#!/bin/bash
# build_ffmp_vr_jars.sh — Build one MOEAFramework problem JAR per N in
# FFMP_VR_N_SWEEP. Each JAR registers `drb_ffmp_N` with the correct
# numberOfVariables for that N (DV count depends on how many interpolated
# NYC/NJ factors fall below the unconstrained threshold).
#
# Must be run from the project root (auto-cd below). Requires:
#   - activated venv with NYCOptimization + pywrdrb importable
#   - javac + jar in PATH
#
# Usage:
#   bash slurm/build_ffmp_vr_jars.sh
#
# Idempotent — safe to re-run after changing FFMP_VR_N_SWEEP or
# generate_ffmp_formulation.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MOEA_DIR="MOEAFramework-5.0"
NATIVE_DIR="${MOEA_DIR}/native"
LIB_DIR="${MOEA_DIR}/lib"

# Expose the Python env so we can query get_n_vars(). The caller is expected
# to have already `module load`ed Python and activated venv, but we double-
# check the import works before looping.
python3 -c "from src.formulations import get_n_vars" >/dev/null 2>&1 || {
    echo "ERROR: cannot import src.formulations. Activate venv first (e.g. 'source venv/bin/activate')." >&2
    exit 1
}

# Read the sweep list from config.py so this script stays in lock-step.
N_SWEEP=$(python3 -c "from config import FFMP_VR_N_SWEEP; print(' '.join(str(n) for n in FFMP_VR_N_SWEEP))")
echo "Building JARs for FFMP_VR_N_SWEEP = [${N_SWEEP// /, }]"

for N in ${N_SWEEP}; do
    NAME="drb_ffmp_${N}"
    NVARS=$(python3 -c "from src.formulations import get_n_vars; print(get_n_vars(f'ffmp_${N}'))")
    NOBJS=$(python3 -c "from src.formulations import get_n_objs; print(get_n_objs())")
    DIR="${NATIVE_DIR}/${NAME}"

    echo ""
    echo "=== ${NAME}: nvars=${NVARS}, nobjs=${NOBJS} ==="

    # Fresh directory
    rm -rf "${DIR}"
    mkdir -p "${DIR}/src/${NAME}" "${DIR}/META-INF/services" "${DIR}/bin"

    # Python stub — required to exist by ExternalProblem, never called at
    # metrics time (MOEAFramework only uses the Java getNumberOfVariables).
    cat > "${DIR}/${NAME}.py" <<PYEOF
import sys

nvars = ${NVARS}
nobjs = ${NOBJS}
nconstrs = 0

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

    # Java Problem class
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

    # Provider class
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

    # Service registration
    echo "${NAME}.${NAME}Provider" > "${DIR}/META-INF/services/org.moeaframework.core.spi.ProblemProvider"

    # Compile
    cp -r "${DIR}/META-INF" "${DIR}/bin/"
    javac -classpath "${LIB_DIR}/*:${DIR}" \
          -d "${DIR}/bin" \
          "${DIR}/src/${NAME}/${NAME}.java" \
          "${DIR}/src/${NAME}/${NAME}Provider.java"

    # Copy Python stub next to compiled class so ExternalProblem can extract it
    mkdir -p "${DIR}/bin/${NAME}"
    cp "${DIR}/${NAME}.py" "${DIR}/bin/${NAME}/"

    # Package
    jar -cf "${DIR}/${NAME}.jar" -C "${DIR}/bin" META-INF/ -C "${DIR}/bin" "${NAME}"
    cp "${DIR}/${NAME}.jar" "${LIB_DIR}/"

    echo "  -> ${LIB_DIR}/${NAME}.jar"
done

echo ""
echo "=== All ${NAME%_*}_N JARs built ==="
ls -la "${LIB_DIR}/drb_ffmp_"*.jar
