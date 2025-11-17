#!/bin/bash
# Show queued jobs with their relative positions in the global priority queue

set -uo pipefail

USER_JOBS="$(squeue -u "$USER" -t PD,R -h -o "%i %P %j %u %T %M %D %R")"

# Read all jobs into an array
mapfile -t lines <<< "$USER_JOBS"

# Prepare default column widths (minimum header size)
declare -A maxlen=( ["POSITION"]=8 ["JOBID"]=5 ["PARTITION"]=9 ["NAME"]=4 ["USER"]=4 ["STATE"]=5 ["TIME"]=4 ["NODES"]=5 ["REASON"]=6 )

# Build format string based on current maxlen
build_format() {
    local fmt=""
    for field in POSITION JOBID PARTITION NAME USER STATE TIME NODES REASON; do
        fmt+="%-$((${maxlen[$field]} + 2))s"
    done
    fmt+="\n"
    echo "$fmt"
}

# Print header
print_header() {
    local fmt
    fmt=$(build_format)
    printf "$fmt" POSITION JOBID PARTITION NAME USER STATE TIME NODES REASON
}

# If there are no pending jobs, print only the header and exit
if [[ ${#lines[@]} -eq 0 || ( ${#lines[@]} -eq 1 && -z "${lines[0]}" ) ]]; then
    print_header
    exit 0
fi

# Build JOBID→POSITION map from sprio (sorted by priority descending)
declare -A pos_map=()
while read -r pos jid; do
    pos_map["$jid"]="$pos"
done < <(sprio | sort -k4 -nr | awk 'NR==1{next} {print $1}' | nl -w1 -s' ')

# First pass: compute max column widths
for line in "${lines[@]}"; do
    [[ -z "$line" ]] && continue
    read -r jobid partition name user state time nodes reason <<< "$line"
    pos="${pos_map[$jobid]:-0}"
    declare -A vals=(
        ["POSITION"]="$pos" ["JOBID"]="$jobid" ["PARTITION"]="$partition"
        ["NAME"]="$name"   ["USER"]="$user"   ["STATE"]="$state"
        ["TIME"]="$time"   ["NODES"]="$nodes" ["REASON"]="$reason"
    )
    for field in "${!vals[@]}"; do
        len=${#vals[$field]}
        (( len > maxlen[$field] )) && maxlen[$field]=$len
    done
done

# Print header (after width computation)
print_header

# Build format for printing rows
fmt=$(build_format)

# Print each queued job
for line in "${lines[@]}"; do
    [[ -z "$line" ]] && continue
    read -r jobid partition name user state time nodes reason <<< "$line"
    pos="${pos_map[$jobid]:-0}"
    printf "$fmt" "$pos" "$jobid" "$partition" "$name" "$user" "$state" "$time" "$nodes" "$reason"
done