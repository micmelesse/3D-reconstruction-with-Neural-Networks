
PORT=6006
PWD=$(pwd)
for d in models_remote/* ; do
    sh scripts/tensorboard_arg.sh $PWD/$d $PORT &
    open http://localhost:$PORT/
    PORT=$((PORT + 1))
done