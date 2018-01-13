batch_size, time_steps, n_layers = 1, 24, 8

# debug loop for a single input in a batch
for b in range(batch_size):
    batch_dir = os.path.join(out_root_dir, "batch_{}".format(b))
    os.makedirs(batch_dir) if not os.path.isdir(
        batch_dir) else print("{} exists".format(batch_dir))

    # encoding layers
    for l in range(n_layers):
        out_dir = os.path.join(batch_dir, "encoding_layer_{}".format(l))
        os.makedirs(out_dir) if not os.path.isdir(
            out_dir) else print("dir exists")
        for t in range(time_steps):
            if l < 7:
                var = encoder_outputs[l][b, t]
                utils.imsave_multichannel(
                    var.eval(fd), out_dir + "/time_{}.png".format(t))
            else:
                var = encoder_outputs[l][b, t].eval(fd)
                plt.figure()
                plt.plot(var)
                plt.savefig(out_dir + "/time_{}.png".format(t))

    # recurrent module
    out_dir = os.path.join(batch_dir, "recurrent_module")
    os.makedirs(out_dir) if not os.path.isdir(out_dir) else print("dir exists")
    for t in range(1):
        val = hidden_state_list[t].eval(fd)
        print(val.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        val = np.random.randint(0, 2, size=[3, 3, 3])
        print(val.shape)

        print(label.shape)
        ax.voxels(label[0], edgecolor='k')

        plt.show()
        plt.close()


       l=mean_loss.eval(fd)
a=mean_accuracy.eval(fd)
print(l,a)

print(np.avg(a))

voxel=tf.argmax(y_hat,axis=4)
voxel=tf.cast(voxel,dtype=tf.float32)
out=voxel.eval(fd)
print(out.shape)

rounded_voxel=tf.round(hidden_state)
out=rounded_voxel.eval(fd)

plt.plot(hidden_state[0,3,0,0,:].eval(fd))

fig=plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(acc[0],edgecolor='k')
ax.view_init(30, 30)
plt.show()
plt.close()

item=30
x=data[item]

y=label[item]

utils.imshow_multichannel(x[0])

 for i in range(4):
            for j in range(4):
                for k in range(4):
                    cell_hidden_state=grid_hidden_state[i,j,k] 
                    plt.figure()
                    plt.plot(cell_hidden_state)
                    plt.savefig(os.path.join(cur_dir,'cell_{}{}{}_at_{}.png'.format(i,j,k,t)))
                    plt.close()
                    
decoder_dir="out/decoder"
# debug decoder net
voxel=tf.argmax(y_hat,axis=4)
voxel=tf.cast(voxel,dtype=tf.float32)

out=voxel.eval(fd)
outvoxel=binvox_rw.Voxels(out,out.shape,[0,0,0],1,'xzy')
with open("out/voxels/{}.binvox".format(),'w') as f:
    outvoxel.write(f)
    
# matplot 3d plot setup
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

%matplotlib inline
feature_vectors=encoder_outputs[7].eval(fd)
plt.figure()
f, axarr = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        t=i*5+j
        if(t<24):
            p=feature_vectors[0,t]
            axarr[i, j].plot(p)

s_input=stacked_input.eval(fd)
print(s_input.shape)
for i in range(4):
    for j in range(4):
        for k in range(4):
            fc=s_input[i,j,k,0,23]
            plt.figure()
            plt.plot(fc)
            plt.savefig(os.path.join("out",'{}{}{}.png'.format(i,j,k)))
            plt.close() 
            sys.exit()
