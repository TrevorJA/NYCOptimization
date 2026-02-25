#!/bin/bash
###############################################################################
# run_diagnostics.sh - MOEA runtime diagnostics using MOEAFramework v5.0
#
# Processes runtime files from MMBorg optimization to compute:
#   1. Per-seed approximation sets (.set files via ResultFileMerger)
#   2. Cross-seed reference set (.ref file via ResultFileSeedMerger)
#   3. Runtime metrics (hypervolume, etc. via MetricsEvaluator)
#
# Prerequisites:
#   - MOEAFramework v5.0 installed (cli executable available)
#   - Java 17+ installed
#   - Runtime files from completed MMBorg runs
#
# Usage:
#   bash run_diagnostics.sh [formulation_name]
#   bash run_diagnostics.sh ffmp
#
# References:
#   - WaterProgramming: "Performing runtime diagnostics using MOEAFramework"
#   - WaterProgramming: "MM Borg Training Part 2"
#   - WaterProgramming: "Introducing MOEAFramework v5.0"
###############################################################################

set -e

FORMULATION=${1:-ffmp}

# --- Paths ---
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPT_DIR="${PROJECT_DIR}/outputs/optimization/${FORMULATION}"
RUNTIME_DIR="${OPT_DIR}/runtime"
SETS_DIR="${OPT_DIR}/sets"
DIAG_DIR="${PROJECT_DIR}/outputs/diagnostics/${FORMULATION}"
REFSET_DIR="${PROJECT_DIR}/outputs/reference_sets"

# MOEAFramework CLI
# TODO: Update path to your MOEAFramework v5.0 installation
MOEA_CLI="${PROJECT_DIR}/tools/MOEAFramework-5.0/cli"

mkdir -p "${SETS_DIR}" "${DIAG_DIR}" "${REFSET_DIR}"

# Read problem dimensions from config
N_OBJS=$(python -c "from config import get_n_objs; print(get_n_objs())")
N_VARS=$(python -c "from config import get_n_vars; print(get_n_vars('${FORMULATION}'))")
EPSILONS=$(python -c "from config import get_epsilons; print(','.join(str(e) for e in get_epsilons()))")

echo "============================================="
echo "MOEA Diagnostics: ${FORMULATION}"
echo "============================================="
echo "Variables: ${N_VARS}, Objectives: ${N_OBJS}"
echo "Epsilons: ${EPSILONS}"
echo "Runtime dir: ${RUNTIME_DIR}"
echo "============================================="


###############################################################################
# Step 1: Extract final approximation sets from runtime files
###############################################################################
echo ""
echo "Step 1: Extracting approximation sets from runtime files..."

RUNTIME_FILES=$(ls ${RUNTIME_DIR}/seed_*_${FORMULATION}.runtime 2>/dev/null)

if [ -z "${RUNTIME_FILES}" ]; then
    echo "ERROR: No runtime files found in ${RUNTIME_DIR}"
    exit 1
fi

for RUNTIME_FILE in ${RUNTIME_FILES}; do
    BASENAME=$(basename "${RUNTIME_FILE}" .runtime)
    SET_FILE="${SETS_DIR}/${BASENAME}.set"

    echo "  Processing: ${BASENAME}"

    ${MOEA_CLI} ResultFileMerger \
        --dimension ${N_OBJS} \
        --output "${SET_FILE}" \
        --epsilon ${EPSILONS} \
        "${RUNTIME_FILE}"
done

echo "  Approximation sets saved to: ${SETS_DIR}"


###############################################################################
# Step 2: Generate cross-seed reference set
###############################################################################
echo ""
echo "Step 2: Generating reference set across all seeds..."

REFSET_FILE="${REFSET_DIR}/${FORMULATION}.ref"

# Merge all seed .set files into a single reference set
# NOTE: Do NOT use --overwrite flag with ResultFileMerger
${MOEA_CLI} ResultFileSeedMerger \
    --dimension ${N_OBJS} \
    --output "${REFSET_FILE}" \
    --epsilon ${EPSILONS} \
    ${SETS_DIR}/seed_*_${FORMULATION}.set

echo "  Reference set: ${REFSET_FILE}"
echo "  Solutions in reference set: $(wc -l < ${REFSET_FILE})"


###############################################################################
# Step 3: Compute runtime metrics (hypervolume, GD, epsilon indicator)
###############################################################################
echo ""
echo "Step 3: Computing runtime metrics..."

for RUNTIME_FILE in ${RUNTIME_FILES}; do
    BASENAME=$(basename "${RUNTIME_FILE}" .runtime)
    METRICS_FILE="${DIAG_DIR}/${BASENAME}.metrics"

    echo "  Evaluating: ${BASENAME}"

    ${MOEA_CLI} MetricsEvaluator \
        --dimension ${N_OBJS} \
        --reference "${REFSET_FILE}" \
        --epsilon ${EPSILONS} \
        --output "${METRICS_FILE}" \
        "${RUNTIME_FILE}"
done

echo "  Metrics saved to: ${DIAG_DIR}"


###############################################################################
# Summary
###############################################################################
echo ""
echo "============================================="
echo "Diagnostics complete for: ${FORMULATION}"
echo "============================================="
echo "Approximation sets:  ${SETS_DIR}/"
echo "Reference set:       ${REFSET_FILE}"
echo "Runtime metrics:     ${DIAG_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/plot_diagnostics.py --formulation ${FORMULATION}"
echo "  2. Check hypervolume convergence across seeds"
echo "  3. Assess reliability (consistency across seeds)"
echo "============================================="
