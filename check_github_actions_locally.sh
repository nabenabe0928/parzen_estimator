run_check() {
    proc_name=${1}
    cmd=${2}
    echo "### Start $proc_name ###"
    echo $cmd
    $cmd
    echo "### Finish $proc_name ###"
    printf "\n\n"
}

target="parzen_estimator"  # target must be modified accordingly
run_check "pytest" "python -m pytest -W ignore"
run_check "black" "black test/ $target/"
