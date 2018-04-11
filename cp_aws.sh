source read_params.sh
rsync -avh --ignore-existing -e "ssh -i ./$KEY" $USER@$DNS:3D-reconstruction-with-neural-networks/out/model* ./aws/
