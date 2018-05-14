source scripts/read_params.sh
rsync -avh -u -e "ssh -i ./$KEY" $USER@$DNS:3D-reconstruction-with-Neural-Networks/models_local/model* ./models_remote/
