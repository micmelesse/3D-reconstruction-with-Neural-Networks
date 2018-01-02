dns=$(cat dns.txt)
ssh -i "thesis.pem" ec2-user@$dns

