#!/bin/bash
CPPCLEAN_RESULTS_FILE_SRC="cppclean_src"
CPPCLEAN_RESULTS_FILE_TEST="cppclean_test"
cppclean ./src/ | grep -v test > ${CPPCLEAN_RESULTS_FILE_SRC};
cppclean ./src/ | grep test > ${CPPCLEAN_RESULTS_FILE_TEST};
NO_WARNINGS=$(wc -l ${CPPCLEAN_RESULTS_FILE_SRC} | awk '{print $1}')
NO_WARNINGS_TEST=$(wc -l ${CPPCLEAN_RESULTS_FILE_TEST} | awk '{print $1}')
echo "Number of warnings:" ${NO_WARNINGS}
echo "Number of warnings in tests:" ${NO_WARNINGS_TEST}
if [ ${NO_WARNINGS} -gt  253 ]; then
    exit 1
else
    exit 0;
fi
