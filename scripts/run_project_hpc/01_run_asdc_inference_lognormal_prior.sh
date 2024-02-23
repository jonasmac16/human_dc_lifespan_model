#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=48gb
#PBS -N JM_DC_Paper_Revision_ASDC_lognormal_prior
#PBS -J 1-30
#PBS -j oe


## Set enviroment var general
export NP=$(wc -l $PBS_NODEFILE | awk '{print $1}')
export WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p' | awk -F: '{ print ($1 * 60) + $2 + ($3 / 60) }')
export JULIA_NUM_THREADS=$NP
export GKSwstype="nul"


## Set enviroment vars project
export PROJECTFOLDER=$HOME/DC_Paper
export SUBFOLDER=notebooks/02_fitting/01_lognormal_prior


## Create input and output folder
mkdir $TMPDIR/PROJECT
mkdir $TMPDIR/JULIA


# copy julia to computing node
scp -r $HOME/JULIA/julia-1.6.1 $TMPDIR/JULIA/.
export julia=$TMPDIR/JULIA/julia-1.6.1/bin/julia
export DEP_PATH=$TMPDIR/JULIA_DEPOT
export julia_hpc_script=${TMPDIR}/PROJECT/scripts/run_project_hpc/intro_hpc.jl
mkdir ${DEP_PATH}
export JULIA_DEPOT_PATH=${DEP_PATH}:$HOME/.julia

# copy folders(project, data, code) to computing node
rsync -a --exclude='notebooks' --exclude='presentations' --exclude='papers' --exclude='experiments' --exclude='_research' --exclude='.vscode' ${PROJECTFOLDER}/ $TMPDIR/PROJECT

## Setup the notebook folder names etc
ARR=()
for d in $PROJECTFOLDER/$SUBFOLDER/*/ ; do
    ARR+=("$d");
done


export JOBFOLDER=$(basename ${ARR[$((PBS_ARRAY_INDEX-1))]})
rsync -a --relative --include="*/" --include="*.jl" --exclude="*" ${PROJECTFOLDER}/./${SUBFOLDER}/${JOBFOLDER} $TMPDIR/PROJECT ## only copy jl files




# setup run command
export run_inference="$julia -t $NP -L ${julia_hpc_script} ${TMPDIR}/PROJECT/${SUBFOLDER}/${JOBFOLDER}/${JOBFOLDER}.jl"


# call script
cd $TMPDIR/PROJECT
${run_inference}


# copy results folder back
mkdir ${PROJECTFOLDER}/${SUBFOLDER}/${JOBFOLDER}/results
rsync -a ${TMPDIR}/PROJECT/${SUBFOLDER}/${JOBFOLDER}/results/ ${PROJECTFOLDER}/${SUBFOLDER}/${JOBFOLDER}/results