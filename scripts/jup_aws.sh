source jup.param
source aws.params
ssh -i ./thesis.pem -L 8157:127.0.0.1:$port $user@$dns