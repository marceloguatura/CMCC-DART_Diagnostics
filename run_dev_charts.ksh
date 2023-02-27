#!/bin/ksh

start=$(date)
echo ""
echo "SERVICE START ${start}"
PATH="/work/csp/mg20022/.conda/envs/guatura/bin"
HOMEPATH="/work/csp/mg20022/charts/CESM-DART/DEV"

VERSION="v1.13"

cd ${HOMEPATH}
PORT=8095

# streamlit run /work/csp/gc02720/borg/cmcc-suite/charts/v1/Home.py '--server.port=8099' --server.headless false  
${PATH}/python -m streamlit run ${HOMEPATH}/${VERSION}/Home.py --server.port=${PORT} --server.headless false --logger.level=info --server.fileWatcherType none


# debug 
# ${PATH}/python -m streamlit run ${HOMEPATH}/${VERSION}/Home.py --server.port=${PORT} --server.headless false --logger.level=debug


echo "SERVICE STOP ${start}"
echo ""