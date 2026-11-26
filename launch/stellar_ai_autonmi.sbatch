#!/bin/bash
# StellarAI (Princeton): the AUTONMI layer's campaigns against NVIDIA Nemotron 3.5 Lightning on vLLM, with
# knobs/machines/stellar_ai_nemotron.yaml. Checked on a laptop (bash -n, a dry run, fake servers). On the machine
# (2026-11-25) run with bash as an srun step, in the FUSION_MODEL_FLEET way only: never submitted with sbatch,
# FUSION_MODEL_NODES not run. The model servers come one of three ways:
#
#   FUSION_MODEL_NODES=0 (default)  this job starts AUTONMI's jobs/model.sh in the background on its own 8 GPUs:
#                                   the servers and the campaigns on one node (for --sim fixture)
#   FUSION_MODEL_NODES=N            it submits N model jobs (AUTONMI's jobs/submit model, one 8-GPU node each) and
#                                   runs the campaigns and their inner loop here, one GPU per worker, under an MPS
#                                   daemon it starts and stops (FUSION_MPS=0: none)
#   FUSION_MODEL_FLEET=DIR          it reuses the servers of an earlier launch's fleet folder (one run with
#                                   FUSION_MODEL_NODES and FUSION_KEEP_SERVERS=1), if each is idle
#
# Export FUSION_MINIAPP_DIR (this checkout, absolute), FUSION_ENV (the campaign's venv), FUSION_SCRATCH (your GPFS
# scratch directory) and RASCAL_REPO (the AUTONMI checkout, whose jobs/local.env sources
# launch/stellar_ai_model.env); with FUSION_MODEL_NODES also FUSION_MODEL_TIME, the model jobs' --time (this job's
# plus a margin, at most 24:00:00). FUSION_RUN_NAME names the run folder in place of the job name: needed when the
# script runs more than once in one job. Export SBATCH_ACCOUNT and SBATCH_PARTITION, or give -A and -p. Then:
#
#   sbatch --time=01:00:00 --output="$FUSION_SCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/stellar_ai_autonmi.sbatch" \
#       --agents 100 --sim fixture
#
# On one node model.sh starts its 8 servers one at a time (about 2 minutes each in an earlier deployment there, so
# about 16 minutes), and Slurm's warning comes 15 minutes before the limit: 30 minutes is too short for that
# (model.sh not yet timed with this model; a stand-in starting all 8 at once took 136 s and 146 s on 2026-11-25).
#
# Its arguments go to `python -m fusion_miniapp.layer_autonmi` after `--preset hundredk` and the two StellarAI
# knob files (so a --knobs file given here wins), then come --llm openai, the fleet's endpoints and key files,
# --model-wait 3600 and --out. FUSION_DRY_RUN=1 prints what it would do and runs nothing (no Slurm needed).
#
# What it does, in order: a run folder $FUSION_SCRATCH/fusion-miniapp/<run name>-<job id> (FUSION_RUN_NAME, else
# the job name; this script's PID in launch.pid there; the fleet folder fleet/ in it, each model job's folder
# fleet/model-<k>/); node-local temporary files (TMPDIR) named for the run; the open-files limit raised to the
# hard limit; the venv activated; one key for every server (fleet/api-key, readable by the owner only, copied
# into each model job's folder before it starts, never printed); the servers
# started and waited for (FUSION_MODEL_WAIT, 3600 s); fleet/endpoints written from the model jobs' own, in order,
# and rewritten whenever one changes (a requeued job; also on a reused fleet); fleet/setup-files, the files a run
# with record.extras copies into its record (and nothing else from these folders); launch/probe_model.py, whose
# failure stops the launch, run in the background and waited for, so that a signal during the check ends the launch
# at once (exit 143; configured and checked on a laptop only); the run, started directly so its PID is the Python
# process's. Then it stops the run only by SIGTERM to that PID: on Slurm's signal 15 minutes before the time limit
# (--signal=B:TERM@900; the run needs about a minute to write its interrupted record) or once the node's available
# memory is below FUSION_MEM_FLOOR_GIB (100). At the end, whatever happened: when this launch started the servers, the stop file in
# each model job's folder, and a model job not yet serving (not running, without its endpoints, or with no state
# from squeue: it would remove its stop file when it starts) cancelled; with FUSION_KEEP_SERVERS=1 (and FUSION_MODEL_NODES) servers that passed
# the check are left running instead, for a later launch's FUSION_MODEL_FLEET, until FUSION_MODEL_TIME or
# `touch DIR/model-*/stop`. Then the MPS daemon quit, the GPUs checked for processes left (when the servers outlive
# this launch, a reused fleet or FUSION_KEEP_SERVERS=1, their VLLM::EngineCore processes are counted, not warned
# about), the node-local folder removed. It exits with the run's status.
#
# To stop a run early: scancel --batch --signal=TERM <job id>, which signals this script alone, as Slurm's warning
# does. A plain scancel sends SIGTERM to every process of the job at once and SIGKILL after Slurm's KillWait, which
# can come before the run has written its record and before the end above: the model jobs then serve until
# FUSION_MODEL_TIME. (Inferred from Slurm's documentation; not yet run on the machine.) Run with bash inside an
# allocation, as on the machine, stop it with kill -TERM $(cat <run folder>/launch.pid) on the node.
#
# FUSION_RUNTIME=dragon (default local) runs the layer under Dragon: FUSION_ENV must then be a venv with Dragon and
# RaDEX's client (README, "What --runtime dragon needs"). The run is `python -m dragon -s
# <checkout>/src/fusion_miniapp/layer_autonmi.py <the same arguments> --runtime dragon`, which the check gets too;
# DRAGON_DEFAULT_SEG_SZ is exported, 64 GiB unless set; <run folder>/run is made with one mkdir before anything
# starts, so a second launch on the same run folder, even one started at the same moment, is refused (exit 2), and
# it is removed at the very end when it is empty and Dragon never started; the signals go to the PID the layer
# writes to <run folder>/run/head.pid, not to Dragon's launcher (one that comes from just before the launcher starts
# until that file appears is held, and sent once it has); the wait for that file is at most 300 s, and at most 10 s
# more once a signal is held; with no PID by then and the launcher still running, the launcher gets SIGTERM, then
# SIGKILL 3 s later, and the script exits 143 if a signal was held, else 1; the exit status is the run record's
# (pass 0, interrupted 143, else 1); and once Dragon has started, dragon-cleanup runs at the end, before the MPS
# daemon quits. Configured and checked on a laptop only (a dry run, and a fake Dragon launcher): no launch through
# this script has run under Dragon on the machine.
#
# FUSION_LAUNCHES=k (default 1, which changes nothing) runs k local launches side by side, as the runs by hand did
# on 2026-11-25: launch i gets --slice i/k (the i-th block of the campaigns, a k-th of workers.sim_workers; with
# --workers W among the arguments, the automatic threads per worker are the CPUs over k x populations x W), block i
# of fleet/endpoints in <run folder>/launch-<i>/endpoints (k must divide the servers, and a server listed twice,
# trailing slashes aside, stops the launch; rewritten when fleet/endpoints changes, except to such a list, which
# leaves the blocks as they were) and --out <run folder>/run-<i>. The check runs once, on the whole fleet (with
# --slice 0/k, so a k that does not divide the campaigns stops the launch there), and one MPS daemon serves every
# launch. A signal goes to each launch still running. Once all have ended, python -m fusion_miniapp.merge_records
# writes one record of them to <run folder>/run; a merge that fails prints a warning and does not change the exit
# status (0 when every launch passed, else a failure's status before 143). Refused (exit 2) with
# FUSION_RUNTIME=dragon, with --runtime dragon (or --runtime=dragon) or --slice among the arguments, and when <run
# folder>/run, run-<i> or launch-<i> exists; <run folder>/run is made as under Dragon, and removed at the very end
# when it is empty and the launches did not get to the merge. Configured and checked on a laptop only (a dry run,
# and launches against fake servers): not run on the machine.
#SBATCH --job-name=fusion-autonmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=950000M
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@900

x() {   # run a command, or with FUSION_DRY_RUN=1 print it
	if [[ $dry == 1 ]]; then echo "[dry run] $(printf '%q ' "$@")"; else "$@"; fi
}

say() {
	echo "[fusion-autonmi $(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
}

new_key() {   # one key for every server, readable by the owner only; never printed
	if [[ $dry == 1 ]]; then say "[dry run] write a new key to $1 (owner only)"; return; fi
	(umask 077 && python -c 'import secrets; print(secrets.token_hex(24))' > "$1")
}

combine() {   # the model jobs' endpoints, in order, into fleet/endpoints (replaced only when changed); false until each has one
	local k tmp="$fleet/endpoints.tmp.$$"
	: > "$tmp"
	for (( k = 0; k < jobs_n; k++ )); do
		[[ -s $fleet/model-$k/endpoints ]] || { rm -f "$tmp"; return 1; }
		cat "$fleet/model-$k/endpoints" >> "$tmp"
	done
	if cmp -s "$tmp" "$fleet/endpoints"; then rm -f "$tmp"; else mv "$tmp" "$fleet/endpoints"; fi
}

alive() {   # the servers' process (on this node) or every model job still there: a job has ended on an end state,
            # or once squeue gave no state 3 times in a row (it prints nothing when it fails, as when slurmctld is slow)
	local id state k=0
	if (( nodes == 0 )); then
		kill -0 "$model_pid" 2>/dev/null && return 0
		say "model.sh ended before its servers were ready: see $fleet/model-0/model-$job.out" >&2
		return 1
	fi
	for id in $model_jobs; do
		state=$(squeue -h -j "$id" -o %T 2>/dev/null) || state=
		case $state in
			BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
				say "model job $id ended ($state) before its servers were ready: see $fleet/model-*/model-$id.out" >&2
				return 1 ;;
			'')
				silent[k]=$(( ${silent[k]:-0} + 1 ))
				if (( silent[k] >= 3 )); then
					say "model job $id: no state from squeue 3 times in a row, taken as ended: see $fleet/model-*/model-$id.out" >&2
					return 1
				fi ;;
			*) silent[k]=0 ;;
		esac
		k=$(( k + 1 ))
	done
}

wait_for_servers() {   # every model job's endpoints, combined; false after FUSION_MODEL_WAIT seconds
	local end=$(( SECONDS + ${FUSION_MODEL_WAIT:-3600} ))
	until combine; do
		alive || return 1
		if (( SECONDS > end )); then
			say "not every model job wrote its endpoints within ${FUSION_MODEL_WAIT:-3600} s" >&2
			return 1
		fi
		sleep 10
	done
}

setup_list() {   # the files a run with record.extras copies (layer_autonmi.SETUP_LIST): these, and nothing else
	local k
	echo endpoints
	for (( k = 0; k < jobs_n; k++ )); do
		printf "model-$k/%s\n" endpoints gpu.csv 'vllm-*.log' 'model-*.out'
	done
}

split_endpoints() {   # FUSION_LAUNCHES: block i of fleet/endpoints, in order, into launch-<i>/endpoints (replaced only when changed)
	local i n tmp twice
	n=$(grep -c '[^[:space:]]' "$fleet/endpoints" 2> /dev/null) || n=0
	if (( n == 0 || n % launches )); then
		say "FUSION_LAUNCHES=$launches must divide the $n servers of $fleet/endpoints" >&2
		return 1
	fi
	twice=$(grep '[^[:space:]]' "$fleet/endpoints" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s:/*$::' | sort | uniq -d)
	if [[ -n $twice ]]; then   # one server in two launches; the blocks stay as they were
		say "$fleet/endpoints names a server more than once (trailing slashes aside): $(echo $twice)" >&2
		return 1
	fi
	for (( i = 0; i < launches; i++ )); do
		tmp="$run_dir/launch-$i/endpoints.tmp"
		grep '[^[:space:]]' "$fleet/endpoints" | sed -n "$(( i * n / launches + 1 )),$(( (i + 1) * n / launches ))p" > "$tmp"
		if cmp -s "$tmp" "$run_dir/launch-$i/endpoints"; then rm -f "$tmp"; else mv "$tmp" "$run_dir/launch-$i/endpoints"; fi
	done
}

mem_guard() {   # SIGTERM to the run (with FUSION_LAUNCHES, to this script, for every launch) once the node's available memory falls below the floor
	local floor=$(( ${FUSION_MEM_FLOOR_GIB:-100} * 1024 * 1024 )) avail
	[[ -r /proc/meminfo ]] || return 0
	while sleep 5; do
		avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
		if (( avail < floor )); then
			say "available memory $(( avail / 1048576 )) GiB, below ${FUSION_MEM_FLOOR_GIB:-100}: SIGTERM to the run ($1)" >&2
			kill -TERM "$1" 2>/dev/null
			return 0
		fi
	done
}

record_status() {   # the exit status a run record gives, as the layer's own: pass 0, interrupted 143, else (or none) 1
	python -c 'import json, sys
try:
    status = json.load(open(sys.argv[1])).get("status")
except (OSError, ValueError):
    status = None
print({"pass": 0, "interrupted": 143}.get(status, 1))' "$1" 2> /dev/null || echo 1
}

on_signal() {   # Slurm's warning (or scancel --batch --signal=TERM): SIGTERM to the run's PID (or each launch's), once; before it starts, the launch ends
	if [[ $checking == starting ]]; then held=1; return 0; fi   # the check is starting: held, then this launch ends
	if [[ -n $starting ]]; then   # under Dragon, before the head has written its PID, or while the launches start: held, then sent
		held=1
		say "signal: held until $starting"
		return 0
	fi
	if [[ -n $pid ]]; then
		if [[ -z $signalled ]]; then
			signalled=1
			say "signal: SIGTERM to the run ($pid)"
			kill -TERM $pid 2>/dev/null || :   # a launch that has already ended is no error
		fi
	else
		say "signal before the run started: stopping" >&2
		exit 143
	fi
}

teardown() {   # always: the servers released if this launch started them, MPS quit, the GPUs checked, node files removed
	local rc=$? k id left mine state cancel
	set +e
	if [[ $rc == 0 && -z $finished ]]; then rc=1; fi   # ended early (bash 3.2 gives an unset variable's exit 0)
	trap - TERM INT
	if [[ $checking =~ ^[0-9]+$ ]]; then   # the check, when a signal ended the launch during it (it writes nothing)
		kill -KILL "$checking" 2>/dev/null; wait "$checking" 2>/dev/null
	fi
	for id in $refresh_pid $guard_pid; do kill "$id" 2>/dev/null; wait "$id" 2>/dev/null; done   # no "Terminated" line
	if (( started && ! keep )); then
		k=0 cancel=
		for id in $model_jobs; do   # a job not yet serving, or whose state squeue does not give, is cancelled: it
			if [[ $dry != 1 ]]; then   # would remove its stop file when it starts (asked before the stop files)
				state=$(squeue -h -j "$id" -o %T 2>/dev/null) || state=
				if [[ $state != RUNNING || ! -s $fleet/model-$k/endpoints ]]; then cancel+=" $id"; fi
			fi
			k=$(( k + 1 ))
		done
		for (( k = 0; k < jobs_n; k++ )); do x touch "$fleet/model-$k/stop"; done
		say "stop files written: the model servers end within seconds"
		for id in $cancel; do x scancel "$id"; done
		if (( nodes == 0 )) && [[ -n $model_pid ]]; then
			for (( k = 0; k < 90; k++ )); do kill -0 "$model_pid" 2>/dev/null || break; sleep 2; done
		fi
	elif (( keep )); then
		say "the model servers are kept for FUSION_MODEL_FLEET=$fleet: they end at FUSION_MODEL_TIME, or on touch $fleet/model-*/stop"
	fi
	if [[ $runtime == dragon && ( $dry == 1 || -n $launcher ) ]]; then   # once Dragon has started: its processes and
		# its memory in /dev/shm on this node, as the runs did
		x timeout 120 dragon-cleanup || say "warning: dragon-cleanup ended with status $?" >&2
		if [[ $dry != 1 && -d /dev/shm ]]; then say "/dev/shm after dragon-cleanup: $(ls /dev/shm | head -5 | tr '\n' ' ')"; fi
	fi
	if (( mps )); then
		if [[ $dry == 1 ]]; then say "[dry run] echo quit | nvidia-cuda-mps-control"; else echo quit | nvidia-cuda-mps-control; fi
	fi
	if [[ $dry == 1 ]]; then
		x nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
	elif command -v nvidia-smi > /dev/null; then
		left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
		mine=$left
		if [[ -n $reuse ]] || (( keep )); then mine=$(grep -v 'VLLM::EngineCore' <<< "$left"); fi
		if [[ -z $left ]]; then say "no process left on the GPUs"
		elif [[ -z $mine ]]; then say "on the GPUs: only the model servers this launch leaves running ($(grep -c . <<< "$left"))"
		else say "warning: still on the GPUs: $mine" >&2; fi
	fi
	x rm -rf "$node_tmp"
	if [[ -n $reserved && -z $launcher && -z $finished ]]; then   # this launch's run/, empty when no Dragon launcher
		rmdir "$reserved" 2>/dev/null                             # started and no merge ran: released last
	fi
	say "ended with status $rc"
	exit "$rc"
}

several() {   # FUSION_LAUNCHES=k > 1, steps 6 and 7: launch i with --slice i/k, block i of the servers and its own folder
              # run-<i>, all at once; each waited for; then their records merged into run (a failed merge only warns)
	local i id s rc=0 dirs=() pids=
	for (( i = 0; i < launches; i++ )); do dirs[i]="$run_dir/run-$i"; done
	if [[ $dry == 1 ]]; then
		for (( i = 0; i < launches; i++ )); do
			x "${run[@]}" "$@" --slice "$i/$launches" --endpoints-file "$run_dir/launch-$i/endpoints" --out "${dirs[i]}"
		done
	else
		starting="every launch has started"   # a signal while they start: held, then sent to each
		for (( i = 0; i < launches; i++ )); do   # argparse keeps the last of a repeated option: these replace the fleet's
			"${run[@]}" "$@" --slice "$i/$launches" --endpoints-file "$run_dir/launch-$i/endpoints" --out "${dirs[i]}" &
			pids+="${pids:+ }$!"
		done
		pid=$pids starting=   # the launches still running, which a signal reaches
		say "the runs' PIDs: $pid"
		if [[ -n $held ]]; then on_signal; fi
		mem_guard "$$" &   # to this script, whose trap signals the launches still running
		guard_pid=$!
		i=0
		for id in $pids; do   # a trapped signal ends a wait early: wait again until the launch has exited
			while :; do
				wait "$id" && s=0 || s=$?
				kill -0 "$id" 2> /dev/null || break
			done
			if [[ $pid == *" "* ]]; then pid=" $pid "; pid=${pid/ $id / }; pid=$(echo $pid); fi   # an ended
			# launch's PID is signalled no more (the last one's stays, as a single run's does)
			say "launch $i/$launches exited with status $s"
			if (( s != 0 )) && [[ $rc == 0 || $rc == 143 ]]; then rc=$s; fi   # a failure before an interruption
			i=$(( i + 1 ))
		done
	fi
	x python -m fusion_miniapp.merge_records "${dirs[@]}" --out "$run_dir/run" ||
		say "warning: the launches' records were not merged into $run_dir/run" >&2
	finished=1
	return "$rc"
}

main() {
	set -euo pipefail
	: "${FUSION_MINIAPP_DIR:?set FUSION_MINIAPP_DIR to this checkout (absolute)}"
	: "${FUSION_ENV:?set FUSION_ENV to the campaign venv (a folder with bin/activate)}"
	: "${FUSION_SCRATCH:?set FUSION_SCRATCH to your GPFS scratch directory}"
	dry=${FUSION_DRY_RUN:-0} nodes=${FUSION_MODEL_NODES:-0} reuse=${FUSION_MODEL_FLEET:-} runtime=${FUSION_RUNTIME:-local}
	started=0 keep=0 mps=0 jobs_n=0 model_pid= model_jobs= pid= signalled= refresh_pid= guard_pid= finished= silent=()
	launcher= starting= held= checking= reserved= launches=${FUSION_LAUNCHES:-1}
	local root name run_dir k id rc probe args run
	root=$(cd "$FUSION_MINIAPP_DIR" && pwd -P)
	job=${SLURM_JOB_ID:-dry} name=${FUSION_RUN_NAME:-${SLURM_JOB_NAME:-fusion-autonmi}}
	run_dir="$FUSION_SCRATCH/fusion-miniapp/$name-$job"
	if [[ ${SLURM_RESTART_COUNT:-0} != 0 ]]; then run_dir+="-restart-$SLURM_RESTART_COUNT"; fi
	fleet=${reuse:-$run_dir/fleet}
	node_tmp="/tmp/fusion-${USER:-user}-${run_dir##*/}"
	if [[ -n $reuse && $nodes != 0 ]]; then
		say "give FUSION_MODEL_FLEET or FUSION_MODEL_NODES, not both" >&2
		return 2
	fi
	if [[ $runtime != local && $runtime != dragon ]]; then
		say "FUSION_RUNTIME is local or dragon, not $runtime" >&2
		return 2
	fi
	if [[ ! $launches =~ ^[1-9][0-9]*$ ]]; then
		say "FUSION_LAUNCHES is a number of launches (1 or more), not $launches" >&2
		return 2
	fi
	if (( launches > 1 )) && [[ $runtime == dragon ]]; then
		say "FUSION_LAUNCHES=$launches with FUSION_RUNTIME=dragon: one Dragon runtime per node and user; give one or the other" >&2
		return 2
	fi
	if (( launches > 1 )); then   # the script's own arguments: its launches run locally, each with the --slice it gives
		local arg before=
		for arg in "$@"; do
			if [[ $arg == --runtime=dragon || ( $before == --runtime && $arg == dragon ) ]]; then
				say "FUSION_LAUNCHES=$launches with --runtime dragon: one Dragon runtime per node and user; give one or the other" >&2
				return 2
			fi
			if [[ $arg == --slice || $arg == --slice=* ]]; then
				say "FUSION_LAUNCHES=$launches gives each launch its --slice: give no --slice" >&2
				return 2
			fi
			before=$arg
		done
	fi
	if [[ ${FUSION_KEEP_SERVERS:-0} == 1 && -z $reuse && $nodes == 0 ]]; then
		say "FUSION_KEEP_SERVERS=1 needs FUSION_MODEL_NODES: model.sh on this node ends with this job" >&2
		return 2
	fi
	if [[ $dry == 1 ]]; then RASCAL_REPO=${RASCAL_REPO:-'$RASCAL_REPO'}; fi   # printed as given, or as its name
	if [[ -z $reuse && $dry != 1 ]]; then
		: "${RASCAL_REPO:?set RASCAL_REPO to the AUTONMI checkout}"
		[[ -f $RASCAL_REPO/jobs/local.env ]] || { say "no $RASCAL_REPO/jobs/local.env: copy local.env.example and source launch/stellar_ai_model.env from it" >&2; return 2; }
	fi
	if (( nodes > 0 )); then : "${FUSION_MODEL_TIME:?set FUSION_MODEL_TIME to the --time of the model jobs}"; fi
	if [[ -n $reuse ]]; then
		say "reusing the servers of $reuse"
	elif (( nodes == 0 )); then
		say "the model servers on this node (model.sh on its GPUs)"
	else
		say "the model servers on $nodes model nodes; the campaigns and their inner loop on this node"
	fi
	say "job $job on $(hostname -s); run folder $run_dir; fleet $fleet"
	if (( launches > 1 )) && [[ $dry != 1 ]]; then   # the records merged are this launch's
		for k in "$run_dir/run" "$run_dir"/run-* "$run_dir"/launch-*; do
			if [[ -e $k ]]; then say "$k exists: with FUSION_LAUNCHES the run folders must be new" >&2; return 2; fi
		done
	fi
	if [[ $dry != 1 && ( $runtime == dragon || $launches != 1 ) ]]; then   # run/ made here, in one step, before anything
		mkdir -p "$run_dir"   # starts: a second launch on this run folder, even one at the same moment, stops here
		if ! mkdir "$run_dir/run" 2> /dev/null; then
			if [[ $runtime == dragon ]]; then say "$run_dir/run exists or cannot be made: under Dragon the run folder must be new" >&2
			else say "$run_dir/run exists or cannot be made: with FUSION_LAUNCHES the run folders must be new" >&2; fi
			return 2
		fi
		reserved=$run_dir/run
	fi
	trap teardown EXIT
	trap on_signal TERM INT

	x mkdir -p "$run_dir" "$node_tmp"
	if [[ $dry == 1 ]]; then say "[dry run] write this script's PID to $run_dir/launch.pid"
	else echo $$ > "$run_dir/launch.pid"; fi   # to stop a launch run with bash: kill -TERM $(cat .../launch.pid)
	export TMPDIR=$node_tmp   # node-local: the inner loop's work folders (sim.work_root) and every temporary file
	x ulimit -n "$(ulimit -Hn)" || say "warning: the open-files limit stays at $(ulimit -n)" >&2   # a model call holds a socket
	set +u   # activation hooks may read unset variables
	x source "$FUSION_ENV/bin/activate"
	set -u

	if [[ -n $reuse ]]; then
		[[ $dry == 1 || ( -s $fleet/endpoints && -s $fleet/api-key && -f $fleet/setup-files ) ]] || {
			say "$fleet is not a fleet folder of this script (endpoints, api-key, setup-files)" >&2; return 2; }
		for k in "$fleet"/model-*/; do [[ -d $k ]] && jobs_n=$(( jobs_n + 1 )); done   # its model jobs, for the refresh
	else
		export RASCAL_REPO
		jobs_n=$(( nodes == 0 ? 1 : nodes ))
		x mkdir -p "$fleet"
		new_key "$fleet/api-key"
		for (( k = 0; k < jobs_n; k++ )); do   # model.sh keeps a key it finds: every server gets this one
			x mkdir -p "$fleet/model-$k"
			x cp -p "$fleet/api-key" "$fleet/model-$k/api-key"
		done
		started=1
		if (( nodes == 0 )); then
			if [[ $dry == 1 ]]; then
				x bash "$RASCAL_REPO/jobs/model.sh" "$fleet/model-0"
			else
				bash "$RASCAL_REPO/jobs/model.sh" "$fleet/model-0" > "$fleet/model-0/model-$job.out" 2>&1 &
				model_pid=$!
			fi
		else
			for (( k = 0; k < jobs_n; k++ )); do
				id=$(x "$RASCAL_REPO/jobs/submit" model "$fleet/model-$k" --parsable --cpus-per-task=96 --mem=950000M \
					--exclusive --time="$FUSION_MODEL_TIME" -J fusion-model)
				[[ $dry == 1 ]] && echo "$id" || say "model job ${id%%;*} for $fleet/model-$k"
				model_jobs+=" ${id%%;*}"
			done
		fi
		if [[ $dry == 1 ]]; then
			say "[dry run] wait for $fleet/model-<k>/endpoints of each of $jobs_n model job(s), then write $fleet/endpoints in order"
		else
			wait_for_servers || return 1
			say "$(wc -l < "$fleet/endpoints" | tr -d ' ') servers in $fleet/endpoints"
		fi
		if [[ $dry == 1 ]]; then
			say "[dry run] write $fleet/setup-files: $(setup_list | tr '\n' ' ')"
		else
			setup_list > "$fleet/setup-files"
		fi
	fi
	if (( launches > 1 )); then   # FUSION_LAUNCHES: each launch its block of the servers, in launch-<i>/ of its own
		if [[ $dry == 1 ]]; then
			say "[dry run] split $fleet/endpoints into $launches blocks in order, $run_dir/launch-<i>/endpoints, each with a setup-files naming it alone"
		else
			for (( k = 0; k < launches; k++ )); do
				mkdir -p "$run_dir/launch-$k"
				echo endpoints > "$run_dir/launch-$k/setup-files"   # what a launch's record.extras copies
			done
			split_endpoints || return 1
		fi
	fi
	if [[ $dry != 1 ]] && (( jobs_n > 0 )); then   # fleet/endpoints follows a requeued job's new ports
		( while sleep 10; do combine || true; if (( launches > 1 )); then split_endpoints || true; fi; done ) > /dev/null 2>&1 &
		refresh_pid=$!
	fi

	args=(--preset hundredk --knobs "$root/knobs/machines/stellar_ai.yaml"
		--knobs "$root/knobs/machines/stellar_ai_nemotron.yaml" "$@" --llm openai
		--endpoints-file "$fleet/endpoints" --api-key-file "$fleet/api-key" --model-wait 3600 --out "$run_dir/run")
	run=(python -m fusion_miniapp.layer_autonmi)
	if [[ $runtime == dragon ]]; then   # the layer as a script under Dragon's launcher, in its single-node mode
		run=(python -m dragon -s "$root/src/fusion_miniapp/layer_autonmi.py")
		args+=(--runtime dragon)   # the check parses it with the layer's own parser
		export DRAGON_DEFAULT_SEG_SZ=${DRAGON_DEFAULT_SEG_SZ:-$(( 64 * 1024 * 1024 * 1024 ))}   # as the run that passed
		say "under Dragon: DRAGON_DEFAULT_SEG_SZ=$DRAGON_DEFAULT_SEG_SZ"
	fi
	probe=(python "$root/launch/probe_model.py" --logs "$fleet")
	if [[ -n $reuse ]]; then probe+=(--idle); fi
	if (( launches > 1 )); then probe+=(--slice "0/$launches"); fi   # the whole fleet; refuses k not dividing the campaigns
	if [[ $dry == 1 ]]; then
		x "${probe[@]}" "${args[@]}"
	else   # in the background, waited for: a signal ends the launch at once, and the teardown ends the check
		checking=starting
		"${probe[@]}" "${args[@]}" &
		checking=$!
		if [[ -n $held ]]; then on_signal; fi   # one that came while it started
		wait "$checking" && rc=0 || rc=$?
		checking=
		if (( rc != 0 )); then say "the model servers failed the check: no run" >&2; return 1; fi
	fi
	if [[ ${FUSION_KEEP_SERVERS:-0} == 1 && -z $reuse ]]; then keep=1; fi   # servers that passed outlive this launch

	if [[ ${FUSION_MPS:-1} != 0 && ( -n $reuse || $nodes != 0 ) ]]; then   # the inner loop has this node's GPUs
		export CUDA_MPS_PIPE_DIRECTORY=$node_tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=$node_tmp/mps/log
		x mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
		x env -u CUDA_VISIBLE_DEVICES nvidia-cuda-mps-control -d   # one daemon for the node's GPUs; workers inherit the pipe
		mps=1
	fi

	if (( launches > 1 )); then several "${args[@]}"; return; fi   # every launch inherits the MPS daemon's pipe
	if [[ $dry == 1 ]]; then
		x "${run[@]}" "${args[@]}"
		finished=1
		return 0
	fi
	if [[ $runtime == dragon ]]; then   # signal the layer's head process (its head.pid), not Dragon's launcher
		starting="the run's head process has written its PID"   # from before the launcher starts: held, then sent
		"${run[@]}" "${args[@]}" &
		launcher=$!
		say "Dragon's launcher: $launcher"
		local limit=600   # 300 s; once a signal is held, 10 s more at most
		for (( k = 0; k < limit; k++ )); do
			[[ -s $run_dir/run/head.pid ]] && break
			kill -0 "$launcher" 2> /dev/null || break
			if [[ -n $held ]] && (( limit > k + 20 )); then limit=$(( k + 20 )); fi
			sleep 0.5
		done
		pid=$(cat "$run_dir/run/head.pid" 2> /dev/null) || pid=
		[[ $pid =~ ^[1-9][0-9]*$ ]] || pid=
		if [[ -z $pid ]] && kill -0 "$launcher" 2> /dev/null; then   # no head to signal: the launcher is ended
			say "no head PID in $run_dir/run/head.pid: ending Dragon's launcher (SIGTERM, and SIGKILL after 3 s)" >&2
			kill -TERM "$launcher" 2> /dev/null || :
			for (( k = 0; k < 6; k++ )); do kill -0 "$launcher" 2> /dev/null || break; sleep 0.5; done
			kill -KILL "$launcher" 2> /dev/null || :
			wait "$launcher" 2> /dev/null || :
			if [[ -n $held ]]; then say "a stop was pending"; return 143; fi
			say "no stop was pending"
			return 1
		fi
		starting=
	else
		"${run[@]}" "${args[@]}" &
		launcher=$! pid=$!
	fi
	if [[ -n $pid ]]; then   # (none under Dragon when the launcher ended before its head wrote one)
		say "the run's PID: $pid"
		if [[ -n $held ]]; then on_signal || :; fi   # a signal held while the head started (set -e: a head gone is no error)
		mem_guard "$pid" &
		guard_pid=$!
	fi
	while :; do   # a trapped signal ends a wait early: wait again until the run has exited
		wait "$launcher" && rc=0 || rc=$?
		kill -0 "$launcher" 2> /dev/null || break
	done
	say "the run exited with status $rc"
	if [[ $runtime == dragon ]]; then   # the record decides, as the layer's own exit status does
		rc=$(record_status "$run_dir/run/run_record.json")
		say "its record gives status $rc"
	fi
	finished=1
	return "$rc"
}

main "$@"; exit
