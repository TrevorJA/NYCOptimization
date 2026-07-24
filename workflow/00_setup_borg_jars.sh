#!/bin/bash
# Step 0 (setup, login node): Build MOEAFramework problem JARs for the active
# formulation set and active objective count.
#
# Builds one JAR per formulation in `PRODUCTION_FORMULATIONS` (or a
# user-supplied list), reading the per-formulation DV count from
# `src.formulations.get_n_vars(name)` and the objective count from
# `src.formulations.get_n_objs()`. Keeps the JARs in lock-step with the
# active config (e.g., salinity-on adds an obj; salt-front DVs add DVs).
#
# Constraint count is DELIBERATELY 0 even though the Borg search itself uses
# formal constraints (src.formulations.get_n_constrs() == 3): every file this
# problem definition parses — solveMPI runtime snapshots and .set files — is
# written via BORG_Archive_append semantics (feasible solutions only, columns
# = variables + objectives, never constraints). Declaring constraints here
# would make MOEAFramework expect constraint columns that never exist in the
# data. Revisit only if allEvaluations output or the serial Borg.solve()
# runtime writer (both of which DO emit constraint columns) is ever parsed.
#
# Usage (from repo root):
#   bash workflow/00_setup_borg_jars.sh                # use PRODUCTION_FORMULATIONS
#   bash workflow/00_setup_borg_jars.sh ffmp ffmp_8    # explicit list
#   NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env bash workflow/00_setup_borg_jars.sh
#
# Idempotent — safe to re-run after changing config knobs.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

MOEA_DIR="MOEAFramework-5.0"
NATIVE_DIR="${MOEA_DIR}/native"
LIB_DIR="${MOEA_DIR}/lib"

# Verify Python imports work before looping.
python3 -c "from src.formulations import get_n_vars, get_n_objs" >/dev/null 2>&1 || {
    echo "ERROR: cannot import src.formulations. Activate venv first." >&2
    exit 1
}

# Resolve the formulation list: explicit args > PRODUCTION_FORMULATIONS.
if [[ $# -gt 0 ]]; then
    FORMULATIONS=("$@")
else
    PROD=$(python3 -c "from config import PRODUCTION_FORMULATIONS; print(' '.join(PRODUCTION_FORMULATIONS))")
    # shellcheck disable=SC2206
    FORMULATIONS=(${PROD})
fi

NOBJS=$(python3 -c "from src.formulations import get_n_objs; print(get_n_objs())")
echo "[00_setup_borg_jars] formulations: ${FORMULATIONS[*]}"
echo "[00_setup_borg_jars] objectives:   ${NOBJS}"

for FORMULATION in "${FORMULATIONS[@]}"; do
    NAME="drb_${FORMULATION}"
    NVARS=$(python3 -c "from src.formulations import get_n_vars; print(get_n_vars('${FORMULATION}'))")
    DIR="${NATIVE_DIR}/${NAME}"

    echo ""
    echo "=== ${NAME}: nvars=${NVARS}, nobjs=${NOBJS} ==="

    rm -rf "${DIR}"
    mkdir -p "${DIR}/src/${NAME}" "${DIR}/META-INF/services" "${DIR}/bin"

    # Python stub — required to exist by ExternalProblem, never called at
    # metrics time (MOEAFramework only uses the Java getNumberOfVariables).
    cat > "${DIR}/${NAME}.py" <<PYEOF
import sys

nvars = ${NVARS}
nobjs = ${NOBJS}
nconstrs = 0  # deliberately 0: parsed files are feasible-only vars+objs (see header)

def evaluate(vars):
    return ([0.0]*nobjs, [0.0]*nconstrs)

if __name__ == "__main__":
    for line in sys.stdin:
        vars = list(map(float, line.split()))
        if len(vars) != nvars:
            sys.exit(f"Incorrect number of variables (expected: {nvars}, actual: {len(vars)})")
        (objs, constrs) = evaluate(vars)
        print(" ".join(map(str, objs + constrs)), flush=True)
PYEOF

    cat > "${DIR}/src/${NAME}/${NAME}.java" <<JEOF
package ${NAME};

import java.io.IOException;
import java.io.UncheckedIOException;

import org.moeaframework.core.Settings;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.ExternalProblem;
import org.moeaframework.problem.ExternalProblem.Builder;
import org.moeaframework.util.io.Resources;
import org.moeaframework.util.io.Resources.ResourceOption;

public class ${NAME} extends ExternalProblem {

    public static final String SCRIPT;

    static {
        try {
            SCRIPT = Resources.asFile(${NAME}.class, "${NAME}.py",
                ResourceOption.REQUIRED, ResourceOption.TEMPORARY).getPath();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to locate executable", e);
        }
    }

    public ${NAME}() {
        super(new Builder().withCommand(Settings.getPythonCommand(), SCRIPT));
    }

    @Override
    public String getName() {
        return "${NAME}";
    }

    @Override
    public int getNumberOfVariables() {
        return ${NVARS};
    }

    @Override
    public int getNumberOfObjectives() {
        return ${NOBJS};
    }

    @Override
    public int getNumberOfConstraints() {
        // Deliberately 0: the runtime/.set files this problem parses are
        // feasible-only vars+objs rows with no constraint columns (see the
        // header comment in 00_setup_borg_jars.sh).
        return 0;
    }

    @Override
    public Solution newSolution() {
        Solution solution = new Solution(getNumberOfVariables(), getNumberOfObjectives(), getNumberOfConstraints());
        for (int i = 0; i < getNumberOfVariables(); i++) {
            solution.setVariable(i, new RealVariable(-1000000.0, 1000000.0));
        }
        return solution;
    }
}
JEOF

    cat > "${DIR}/src/${NAME}/${NAME}Provider.java" <<PEOF
package ${NAME};

import org.moeaframework.core.spi.RegisteredProblemProvider;

public class ${NAME}Provider extends RegisteredProblemProvider {

    public ${NAME}Provider() {
        super();
        register("${NAME}", ${NAME}::new, null);
        registerDiagnosticToolProblems(getRegisteredProblems());
    }
}
PEOF

    echo "${NAME}.${NAME}Provider" > "${DIR}/META-INF/services/org.moeaframework.core.spi.ProblemProvider"

    cp -r "${DIR}/META-INF" "${DIR}/bin/"
    javac -classpath "${LIB_DIR}/*:${DIR}" \
          -d "${DIR}/bin" \
          "${DIR}/src/${NAME}/${NAME}.java" \
          "${DIR}/src/${NAME}/${NAME}Provider.java"

    mkdir -p "${DIR}/bin/${NAME}"
    cp "${DIR}/${NAME}.py" "${DIR}/bin/${NAME}/"

    jar -cf "${DIR}/${NAME}.jar" -C "${DIR}/bin" META-INF/ -C "${DIR}/bin" "${NAME}"
    cp "${DIR}/${NAME}.jar" "${LIB_DIR}/"

    echo "  -> ${LIB_DIR}/${NAME}.jar"
done

echo ""
echo "=== All JARs built ==="
ls -la "${LIB_DIR}/"*.jar
