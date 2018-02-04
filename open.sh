#!/bin/bash

if [ -f jupyter_session_* ]
then 
    echo jupyter session is already running
    sh localhost.sh
else
    echo new jupyter session
    jupyter notebook &> jupyter_session_$(date +"%Y-%m-%d_%H:%M:%S").txt&
fi
