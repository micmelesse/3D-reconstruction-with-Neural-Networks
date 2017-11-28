#!/bin/bash
jupyter notebook stop 8888 # this works in the latest version of jupyter
jupyter notebook &> /dev/null &
jupyter notebook list