source params/jup.param
source params/aws.params
ssh -i ./thesis.pem -L 8157:127.0.0.1:$PORT $USER@$DNS