source params/aws.params
scp -ri $KEY $USER@$DNS:3D-reconstruction-with-neural-networks/out/model* ./aws/
