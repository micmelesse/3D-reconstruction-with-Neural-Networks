import os

def save_encoder_state(self, save_dir, fd):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    n_layers = len(self.encoder_outputs)
    for l in range(n_layers):
        state = self.encoder_outputs[l].eval(fd)
        n_batch = state.shape[0]
        for b in range(n_batch):
            n_time = state.shape[1]
            for t in range(n_time):
                np.save(save_dir + "/encoder_{}-{}-{}".format(l,
                                                              b, t), state[b, t])
                if l == n_layers - 1:
                    plt.plot(state[b, t])
                    plt.savefig(
                        save_dir + "/encoder_{}-{}-{}.png".format(l, b, t))
                    plt.close()

                else:
                    utils.imsave_multichannel(
                        state[b, t], save_dir + "/encoder_{}-{}-{}.png".format(l, b, t))


def save_decoder_state(self, save_dir, fd):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    n_layers = len(self.decoder_outputs)
    for l in range(n_layers):
        state = self.decoder_outputs[l].eval(fd)
        n_batch = state.shape[0]
        for b in range(n_batch):
            n_channels = state.shape[-1]
            for c in range(n_channels):
                np.save(save_dir + "/decoder_{}-{}-{}".format(l,
                                                              b, c), state[b, :, :, :, c])
                utils.imsave_voxel(state[b, :, :, :, c], save_dir +
                                   "/decoder_{}-{}-{}.png".format(l, b, c))


def save_hidden_state(self, save_dir, fd):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    n_layers = len(self.hidden_state_list)
    for l in range(n_layers):
        state = self.hidden_state_list[l].eval(fd)
        # print(state.shape)
        n_batch = state.shape[0]
        for b in range(n_batch):
            n_channels = state.shape[-1]
            for c in range(n_channels):
                np.save(save_dir + "/hidden_{}-{}-{}".format(l,
                                                             b, c), state[b, :, :, :, c])
                utils.imsave_voxel(state[b, :, :, :, c], save_dir +
                                   "/hidden_{}-{}-{}.png".format(l, b, c))
