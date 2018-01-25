source config/jup.param
source config/aws.params
ssh -i ./thesis.pem -L 8157:127.0.0.1:$PORT $USER@$DNS