#!/bin/bash
set -e
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
BASE="/Applications/NetLogo 7.0.3"
JAR="$BASE/app/netlogo-7.0.3.jar"
MODEL="netlogo/coherence_model_tencon.nlogox"
run_exp () {
  local name="$1"
  "$JAVA_HOME/bin/java" -XX:MaxRAMPercentage=50 -Dfile.encoding=UTF-8 \
    -Dnetlogo.extensions.dir="$BASE/extensions" -Dnetlogo.models.dir="$BASE/models" \
    --add-exports=java.base/java.lang=ALL-UNNAMED --add-exports=java.desktop/sun.awt=ALL-UNNAMED \
    --add-exports=java.desktop/sun.java2d=ALL-UNNAMED --add-exports=java.desktop/com.apple.laf=ALL-UNNAMED \
    -classpath "$JAR" org.nlogo.headless.Main \
    --model "$MODEL" --experiment "$name" \
    --table "exports/${name}-table.csv" --threads 8
}
echo "[$(date '+%H:%M:%S')] START S-H004"
run_exp S-H004_ai_param_robustness
echo "[$(date '+%H:%M:%S')] DONE S-H004 -> $(grep -c '^"' exports/S-H004_ai_param_robustness-table.csv) rows"
echo "[$(date '+%H:%M:%S')] START S-H005"
run_exp S-H005_fatigue_robustness
echo "[$(date '+%H:%M:%S')] DONE S-H005 -> $(grep -c '^"' exports/S-H005_fatigue_robustness-table.csv) rows"
echo "[$(date '+%H:%M:%S')] ALL SWEEPS COMPLETE"
