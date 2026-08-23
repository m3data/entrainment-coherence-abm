#!/bin/bash
set -e
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
BASE="/Applications/NetLogo 7.0.3"; JAR="$BASE/app/netlogo-7.0.3.jar"
MODEL="netlogo/coherence_model_tencon.nlogox"
run_exp () {
  local name="$1"
  "$JAVA_HOME/bin/java" -XX:MaxRAMPercentage=50 -Dfile.encoding=UTF-8 \
    -Dnetlogo.extensions.dir="$BASE/extensions" -Dnetlogo.models.dir="$BASE/models" \
    --add-exports=java.base/java.lang=ALL-UNNAMED --add-exports=java.desktop/sun.awt=ALL-UNNAMED \
    --add-exports=java.desktop/sun.java2d=ALL-UNNAMED --add-exports=java.desktop/com.apple.laf=ALL-UNNAMED \
    -classpath "$JAR" org.nlogo.headless.Main \
    --model "$MODEL" --experiment "$name" \
    --table "exports/seeded/${name}-seeded.csv" --threads 8
}
for exp in H001_batch2_proportion_sweep_full H002_batch3_repeated_stress H003_batch4_mixed_regime S-H004_ai_param_robustness S-H005_fatigue_robustness; do
  echo "[$(date '+%H:%M:%S')] START $exp"
  run_exp "$exp"
  echo "[$(date '+%H:%M:%S')] DONE  $exp -> $(grep -c '^"[0-9]' "exports/seeded/${exp}-seeded.csv") rows"
done
echo "[$(date '+%H:%M:%S')] ALL SEEDED RUNS COMPLETE"
