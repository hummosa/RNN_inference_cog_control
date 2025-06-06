import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import logger
import torch
import matplotlib.transforms as mtransforms

# make the axis lines thinner
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def explore_data_container(data):
    """
    Explores a nested data container (list, numpy array, or PyTorch tensor).
    Args:
        data: The input data container.
    Returns:
        None
    """
    def print_info(layer, depth):
        if isinstance(layer, list):
            print(f"Layer {depth}: List, Length = {len(layer)}")
            for item in layer:
                print_info(item, depth + 1)
                break  # Only print the first item
        elif isinstance(layer, np.ndarray):
            print(f"Layer {depth}: Numpy Array, Shape = {layer.shape}")
        elif isinstance(layer, torch.Tensor):
            print(f"Layer {depth}: PyTorch Tensor, Shape = {tuple(layer.shape)}")
            return  # Stop exploring when a tensor is encountered
        else:
            print(f"Layer {depth}: Unknown type")
    print_info(data, depth=0)


# %%
def plot_behavior(explore_data_container, logger, config, x2=None, print_shapes=False):
    fig, axes = plt.subplot_mosaic([['A', 'A'], ['B', 'B'], ['C', 'C'], ['D', 'D'], ], sharex=False, sharey=False,
                                    constrained_layout=False, figsize = [18/2.53, 18/2.53])

    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    if print_shapes:
        print('logger has the following objects: ')
        print(logger.__dict__.keys())
        print('logger.llcids has: ')
        explore_data_container(logger.llcids)
        print('logger.hlcids has: ')
        explore_data_container(logger.hlcids)
        print('logger.combined_inputs has: ')
        explore_data_container(logger.training_batches)

        print('ci: ')
        print(np.concatenate(logger.training_batches, axis=0).shape)

        print('ll')
        print(np.concatenate(logger.llcids, axis=0).shape)
        print('hh')
        print(np.concatenate(logger.hlcids, axis=0).shape)

    # merge the batches into one sequence
    ci = np.concatenate(logger.training_batches, axis=0)
    ci = ci.reshape(-1, ci.shape[-1])
    #li = ci[:, -config.latent_dims[0]:]  # latent
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])

    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    if logger.predicted_outputs != []:
        oi = np.concatenate(logger.predicted_outputs, axis=0)
        oi = oi.reshape(-1, oi.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, ll.shape[-1])
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1, hh.shape[-1])

    if print_shapes:
        print('ci: ', ci.shape)

    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    hh_colors = hh_cmap(np.arange(len(unique_hh)))

    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('viridis', len(unique_ll))
    ll_colors = ll_cmap(np.arange(len(unique_ll)))

    # x1, x2 = 0, min(5000, ci.shape[0])
    if x2 is None:
        x1, x2 = 0, ii.shape[0]
    else:
        x1, x2 = 0, x2
    
    ax = axes['A'] 
    if (ii.shape[-1] ) > 1: # if there are more than one features
        ax.imshow(ii[x1:x2,].T, aspect='auto', cmap='viridis', interpolation='none')
        ax.set_ylabel('Feature')
    else: # if input is 1D
        ax.plot(ii[x1:x2,], '.', alpha = 0.7, markersize =3, linewidth=1, color=obs_color)
        if logger.predicted_outputs != []:
            ax.plot(oi[x1:x2,], '.', alpha = 0.7, markersize =3, linewidth=1, label='Predicted Output', color=preds_color)
        ax.set_ylabel('Feature')
    # shade the background alternatively using ax span for each block
    if config.experiment_name == 'contextual_switching_task':
        first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
        for i in range(first_block_start_ts+1, x2, config.block_size,):
            if i < ll.shape[0]:
                ax.axvspan(i, i+config.block_size, color=ll_cmap(ll[i][0]), alpha=0.041)
                # ax.text(i+config.block_size/2, 1.1, f'{ll[i][0]:.1f}', fontsize=6, ha='center', va='center', color='black')
                # plot high level latent        
                # ymin = 0  # Specify the minimum y value for the span
                # ymax = 0.05  # Specify the maximum y value for the span
                # ax.axvspan(i, i+config.block_size, ymin=ymin, ymax=ymax, color=hh_cmap(hh[i][0]-1), alpha=0.5)
        #easier and more reliable way to plot hh
        ax.scatter(range(x1, x2), np.zeros(x2-x1), c=hh[x1:x2], cmap=hh_cmap, s=3)
        # ax.scatter(range(x1, x2), np.ones(x2-x1), c=ll[x1:x2], cmap=ll_cmap, s=3)
    ax.set_xticklabels([])
    ax.set_xlim(x1, x2)
    
    ax = axes['B']

    if li.shape[1] > 2:
        ax.imshow(li[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none')
    else: # if latent  is 2D
        ax.plot(li[x1:x2,], alpha = 0.7, linewidth=1,)
    ax.set_xlim(x1, x2)
    ax.set_ylabel('Latent')
    ax.set_xticklabels([])

    ax = axes['C']
    # plot predicted losses
    if logger.prediction_losses != []:
        prediction_losses = np.concatenate(logger.prediction_losses, axis = 0)
        prediction_losses = prediction_losses.reshape(-1, prediction_losses.shape[-1])
        prediction_losses = prediction_losses.mean(axis=-1) # average over the output dimensions
    else:
        prediction_losses = np.concatenate(logger.training_losses, axis = 0)
        prediction_losses = prediction_losses.reshape(-1, prediction_losses.shape[-1])
        prediction_losses = prediction_losses.mean(axis=-1) # average over the output dimensions
    ax.plot(prediction_losses, linewidth=0.5, alpha = 0.7, color='grey', label='Prediction Loss')
    # convolve with a window = 10 to smooth the plot
    ax.plot(np.convolve(prediction_losses.squeeze(), np.ones(10)/10, mode='valid',), linewidth=1, color='black', label='Smoothed Prediction Loss')
    ax.set_ylabel('Prediction Loss')
    ax.set_xlabel('Time step')
    ax.set_xlim(x1, x2)
    if config.experiment_name == 'contextual_switching_task':
        ax.set_ylim(0, 0.6)
    ax.text(0.5, 0.5, f'Total Loss: {prediction_losses.mean():.2f}', fontsize=12, ha='center', va='center', color='black')

    ax = axes['D']
    # plot the output
    if logger.predicted_outputs != [] and (oi.shape[-1] ) > 1:
        ax.imshow(oi[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none')
        ax.set_ylabel('Output')
    else:
        pass

    # try:
    #     ax = axes['D'] # plot the average loss per block
    #     if config.experiment_name == 'contextual_switching_task':
    #         if logger.prediction_losses != []:
    #             prediction_losses = np.concatenate(logger.prediction_losses, axis = 0)
    #         else:
    #             prediction_losses = np.concatenate(logger.training_losses, axis = 0)

    #         timesteps_with_data_lost = config.seq_len -1
    #         index_where_first_block_ends = config.block_size - timesteps_with_data_lost
    #         prediction_losses = prediction_losses[index_where_first_block_ends:]
    #         if config.epochs ==1: # only, cuz otherwise still have not fixed how to calculate how much data to discard for each epoch.
    #             # The problem is the begining seq_len chunk of data initially, I record only the last config.stride elements. So I have to take that initial data out to get the correct alignment of all block data to the block switch points. 

    #             per_block_maybe = prediction_losses.reshape(-1, config.block_size )
    #             # explore_data_container(per_block_maybe)

    #             ax.plot(per_block_maybe.mean(axis=0), label='Average Loss', linewidth=3)
    #             ax.plot(per_block_maybe.T, label='Loss per block', alpha=0.1, linewidth=0.5)
    #         ax.set_xlim(0, 10)
    #     else:
    #         pass

    #     ax = axes['E']
    #     if config.experiment_name == 'contextual_switching_task':
    #         early_losses = per_block_maybe[:, :10].mean(axis=1)
    #         late_losses = per_block_maybe[:, 10:].mean(axis=1)
    #         whole_block_losses = per_block_maybe.mean(axis=1)

    #         ax.boxplot([early_losses, late_losses, whole_block_losses], labels=['Early', 'Late', 'Whole'])
    #         ax.set_ylim(0, 0.1)
    #     else:
    #         total_loss = prediction_losses.mean()
    #         ax.text(0.5, 0.5, f'Total Loss: {total_loss:.2f}', fontsize=12, ha='center', va='center', color='black')
    #         ax.boxplot(prediction_losses, labels=['Total Loss'])
    #         ax.set_ylim(0, 1)
    #     ax.set_ylabel('Loss')
    # except:
    #     pass
    # uncomment to draw low level classes
    # for i, ll_id in enumerate(unique_ll):
    #     ax.plot(ll_id * (ll[x1:x2] == ll_id), '.',linewidth=1,  label=ll_id, color=ll_colors[i])
    # ax.set_ylabel('LL id')
    # ax.set_xlim(x1, x2)
    # ax = axes[3]
    # for i, hh_id in enumerate(unique_hh):
    #     ax.plot(hh_id * (hh[x1:x2] == hh_id), '.',linewidth=1,  label=hh_id, color=hh_colors[i])
    # ax.set_ylabel('HL id')
    # ax.set_xlim(x1, x2)

    # ADDITIONAL PLOTS for possible future use.
    # #%% Find the corr between latent dims and high level latent
    # for i in range(ci.shape[-1]):
    #     correlation = np.corrcoef(ci[:, i].squeeze(), hh.squeeze())[0, 1]
    #     print(f'Correlation between latent dim {i} and high level latent: {correlation:.2f}')

    # fig, axes = plt.subplots(10, 1, figsize=(10, 20))
    # for i in range(10):
    #     ax = axes[i]
    #     ax.plot(ci[:, i], label=f'Latent Dim {i+1}', alpha=0.9, linewidth=.5)
    #     ax.set_ylabel('Latent Dim')
    #     ax.legend()
    #     ax.set_xlim(x1, x2)

    #     unique_hh = np.unique(hh)

    #     hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    #     hh_colors = hh_cmap(np.arange(len(unique_hh)))
    #     for i, hh_id in enumerate(unique_hh):
    #         ax.plot(hh_id * (hh == hh_id), '.',linewidth=.1,  alpha = 0.1,label=hh_id, color=hh_colors[i])

    # plt.tight_layout()
    # plt.show()


    # Plot the latent dims as lines and the high level latent as dots
    # ax = plt.gca()
    # for i in [2,4]:#range(config.latent_dims[0]):
    #     ax.plot(ci[:, i], label=f'Latent Dim {i+1}', alpha=0.3, linewidth=.5)

    # # plot hh as lines
    # unique_hh = np.unique(hh)
    # hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    # hh_colors = hh_cmap(np.arange(len(unique_hh)))
    # for i, hh_id in enumerate(unique_hh):
    #     ax.plot(hh_id * (hh == hh_id), '.',linewidth=.1,  alpha = 0.1,label=hh_id, color=hh_colors[i])

    # ax.set_ylabel('Latent Dim')
    # ax.legend()
    # ax.set_xlim(x1, x2)
    # fig.tight_layout()
    return fig
#%%
# from utils.plot_utils import *

def plot_task_and_hierarchies_illustration(logger,  config, x2=None, show_output=False):
        
    obs_color = 'tab:grey'
    preds_color = 'tab:red'

    fig, axes = plt.subplot_mosaic([['A'], ['B'], ['B']], sharex=True,
                                    constrained_layout=False, figsize = [16/2.53, 4.5/2.53], dpi=300)
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # merge the batches into one sequence
    ci = np.concatenate(logger.training_batches, axis=0)
    ci = ci.reshape(-1, ci.shape[-1])
    #li = ci[:, -config.latent_dims[0]:]  # latent
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])

    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    if logger.predicted_outputs != []:
        oi = np.concatenate(logger.predicted_outputs, axis=0)
        oi = oi.reshape(-1, oi.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, ll.shape[-1])
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1, hh.shape[-1])


    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    # hh_cmap = plt.get_cmap('winter', len(unique_hh))
    hh_colors = hh_cmap(np.arange(len(unique_hh)))

    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('Set1', len(unique_ll))
    ll_colors = ll_cmap(np.arange(len(unique_ll)))

    # x1, x2 = 0, min(5000, ci.shape[0])
    if x2 is None:
        x1, x2 = 0, ii.shape[0]
    else:
        x1, x2 = 0, x2
    
    ax = axes['B'] 
    if (ii.shape[-1] ) > 1: # if there are more than one features
        ax.imshow(ii[x1:x2,].T, aspect='auto', cmap='viridis', interpolation='none')
        ax.set_ylabel('Feature')
    else: # if input is 1D
        ax.plot(ii[x1:x2,], 'o',  markersize =0.5, color=obs_color)
        if show_output:
            ax.plot(oi[x1:x2,], 'o',   markersize =0.5,  label='Predicted Output', color=preds_color)
        ax.set_ylabel('Feature')
    ax.set_xlabel('Time steps')
    # shade the background alternatively using ax span for each block
    if config.experiment_name == 'contextual_switching_task':
        first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
        for i in range(first_block_start_ts+1, x2, config.block_size,):
            if i < ll.shape[0]:
                ax.axvspan(i, i+config.block_size, color=ll_cmap(ll[i][0]), alpha=0.04)

        #easier and more reliable way to plot hh
        # ax.scatter(range(x1, x2), np.zeros(x2-x1), c=hh[x1:x2], cmap=hh_cmap, s=3)
        # ax.scatter(range(x1, x2), np.ones(x2-x1), c=ll[x1:x2], cmap=ll_cmap, s=3)
    ax.set_xlim(x1, x2)
    
    ax = axes['A']
    first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
    most_recent_hh_value = 0
    for i in range(first_block_start_ts+1, x2, config.block_size,):
        # ax.axvline(c, .6, .9 , color=color, linestyle='-', alpha=0.6, linewidth=3)
        if i < ll.shape[0]:
            color = ll_cmap(ll[i][0])
            ax.axvline(i, 0.0,0.4,  alpha=0.6, color=color, linewidth=3)
            if hh[i] != most_recent_hh_value:
                ax.axvline(i, .6, 0.9, color=hh_cmap(hh[i][0]-1), linestyle='-', alpha=0.6, linewidth=3)
                most_recent_hh_value = hh[i]

    ax.set_axis_off()
    ax.set_xlim(x1, x2)
# plot_task_and_hierarchies_illustration(logger, config, x2=1000)
# plt.savefig(f'{config.export_path}_task_and_hierarchies_illustration.pdf', dpi=300, bbox_inches='tight')
# print('figure saved to: ', f'{config.export_path}_task_and_hierarchies_illustration.pdf')
#%%
import numpy as np
from sklearn.manifold import TSNE

# Plot t-SNE results
def plot_tnse(logger, config):
    ci = np.concatenate(logger.training_batches, axis=0)
    ci = ci.reshape(-1, ci.shape[-1])
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])
    # li = ci[:, -config.latent_dims[0]:]  # latent
    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, ll.shape[-1])
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1, hh.shape[-1])

    # trying to debug
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1,)
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1,)

    # Apply t-SNE
    how_many = 10000
    ll = ll[-how_many:]
    hh = hh[-how_many:]
    dimensionality = ii.shape[-1]
    if dimensionality > 1:
        ii_tsne = TSNE(n_components=2).fit_transform(ii[-how_many:, ])
    else:
        ii_tsne = TSNE(n_components=1).fit_transform(ii[-how_many:, ])
    latent_tsne = TSNE(n_components=2).fit_transform(li[-how_many:, ])
    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('prism', len(unique_hh))
    hh_colors = hh_cmap(np.arange(len(unique_hh)))

    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('viridis', len(unique_ll))
    ll_colors = ll_cmap(np.arange(len(unique_ll)))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=100)
    try:
        ax = axes[0, 0]
        ax.set_title('Input, by low level label')
        for i, ll_id in enumerate(unique_ll):
            # ax.scatter three fake data points to make the legend work
            ax.scatter(0, 0, c=ll_cmap(ll_id), label=ll_id, s=9, alpha=0.9)
            ax.legend()

        if dimensionality ==1:
            ax.scatter(ii_tsne[:, ], ii_tsne[:,], c=ll[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
                # ax.scatter(ii_tsne[ll == ll_id], ii_tsne[ll == ll_id], c=ll_colors[i], label=ll_id, s=3, alpha=0.3)
                # ax.scatter(ii_tsne[ll == ll_id], ii_tsne[ll == ll_id], c=ll_cmap(ll_id),  s=3, alpha=0.3)
    
        else:
            ax.scatter(ii_tsne[:, 0], ii_tsne[:, 1], c=ll[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
        # ax.legend()
        ax.set_xlabel('t-SNE Dim 2')
        ax.set_ylabel('t-SNE Dim 1')

        ax = axes[0, 1]
        ax.set_title('by high level label')
        for i, hh_id in enumerate(unique_hh):
            # ax.scatter three fake data points to make the legend work
            ax.scatter(0, 0, c=hh_cmap(hh_id-1), label=hh_id, s=9, alpha=0.9)
            ax.legend()

        if dimensionality ==1:
            ax.scatter(ii_tsne[:, ], ii_tsne[:,], c=(hh[:ii_tsne.shape[0]]), cmap='prism', s=2, alpha=0.4)
        else:
            ax.scatter(ii_tsne[:, 0], ii_tsne[:, 1], c=hh[:ii_tsne.shape[0]], cmap='prism', s=2, alpha=0.4)

        ax.legend()
        ax.set_xlabel('t-SNE Dim 2')
        ax.set_ylabel('t-SNE Dim 1')
    except:
        pass
    # Plot t-SNE results for latent
    ax = axes[1, 0]
    ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=ll[:latent_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('t-SNE Dim 2')
    ax.set_ylabel('t-SNE Dim 1')
    ax.set_title('Latent by low level label')

    ax = axes[1, 1]
    ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=(hh[:latent_tsne.shape[0]]), cmap='prism' ,s=1, alpha=0.5)

    # ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=hh_cmap(hh[:latent_tsne.shape[0]]), s=1, alpha=0.5)
    ax.set_xlabel('t-SNE Dim 2')
    ax.set_ylabel('t-SNE Dim 1')
    ax.set_title('by high level label')

    # label only outer panels and hide inner panels
    for ax in axes.flatten():
        # decrease the font of the xlabel and ylabel
        ax.label_outer()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()

# Plot t-SNE results
def plot_tnse_previous_colors(logger, config):
    ''' for some reason these colors seem more informative of mixing states than the new '''
    ci = np.concatenate(logger.training_batches, axis=0)
    ci = ci.reshape(-1, ci.shape[-1])
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])
    # li = ci[:, -config.latent_dims[0]:]  # latent
    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, ll.shape[-1])
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1, hh.shape[-1])

    # trying to debug
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1,)
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1,)

    # Apply t-SNE
    how_many = 10000
    ll = ll[-how_many:]
    hh = hh[-how_many:]
    dimensionality = ii.shape[-1]
    if dimensionality > 1:
        ii_tsne = TSNE(n_components=2).fit_transform(ii[-how_many:, ])
    else:
        ii_tsne = TSNE(n_components=1).fit_transform(ii[-how_many:, ])
    latent_tsne = TSNE(n_components=2).fit_transform(li[-how_many:, ])

    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    hh_colors = hh_cmap(np.arange(len(unique_hh)))

    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('viridis', len(unique_ll))
    ll_colors = ll_cmap(np.arange(len(unique_ll)))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=100)
    try:
        ax = axes[0, 0]
        ax.set_title('Input, by low level label')
        for i, ll_id in enumerate(unique_ll):
            # ax.scatter three fake data points to make the legend work
            ax.scatter(0, 0, c=ll_cmap(ll_id), label=ll_id, s=7, alpha=0.9)
            ax.legend()

        if dimensionality ==1:
            ax.scatter(ii_tsne[:, ], ii_tsne[:,], c=ll[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
                # ax.scatter(ii_tsne[ll == ll_id], ii_tsne[ll == ll_id], c=ll_colors[i], label=ll_id, s=3, alpha=0.3)
                # ax.scatter(ii_tsne[ll == ll_id], ii_tsne[ll == ll_id], c=ll_cmap(ll_id),  s=3, alpha=0.3)
    
        else:
            ax.scatter(ii_tsne[:, 0], ii_tsne[:, 1], c=ll[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
        # ax.legend()
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 1')

        ax = axes[0, 1]
        ax.set_title('by high level label')
        for i, hh_id in enumerate(unique_hh):
            # ax.scatter three fake data points to make the legend work
            ax.scatter(0, 0, c=hh_cmap(hh_id), label=hh_id, s=7, alpha=0.9)
            ax.legend()

        if dimensionality ==1:
            ax.scatter(ii_tsne[:, ], ii_tsne[:,], c=hh[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
        else:
            ax.scatter(ii_tsne[:, 0], ii_tsne[:, 1], c=hh[:ii_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)

        ax.legend()
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 1')
    except:
        pass
    # Plot t-SNE results for latent
    ax = axes[1, 0]
    ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=ll[:latent_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 1')
    ax.set_title('Latent by low level label')

    ax = axes[1, 1]
    ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=hh[:latent_tsne.shape[0]], cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 1')
    ax.set_title('by high level label')

    # label only outer panels and hide inner panels
    for ax in axes.flatten():
        ax.label_outer()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()

# visuzlize the task dataset, latent and high level latent, and the data obtained from getitem
def plot_dataset_sample(explore_data_container, config, task_dataset):
    fig, axes = plt.subplots(3, 1, figsize=(8, 5))
    dataloader = DataLoader(task_dataset, batch_size=config.batch_size, shuffle=False)

    data, latent, high_level_latent = next(iter(dataloader))

    explore_data_container(data)
    print("Showing data")

    explore_data_container(latent)
    print("Showing latent")

    explore_data_container(high_level_latent)
    print("Showing high level latent")

    concat_data = np.stack (data.cpu().detach().numpy(), axis=0).reshape(-1, data.shape[-1])
    concat_latent = np.stack (latent.cpu().detach().numpy(), axis=0).reshape(-1, 1)
    concat_high_level_latent = np.stack (high_level_latent.cpu().detach().numpy(), axis=0).reshape(-1, 1)

    # uncomment to view enitre dataset
    # concat_data = np.stack (task_dataset.data_sequence, axis=0).reshape(-1, 1)
    # concat_latent = np.stack (task_dataset.latent_sequence, axis=0).reshape(-1, 1)
    # concat_high_level_latent = np.stack (task_dataset.high_level_latent_sequence, axis=0).reshape(-1, 1)

    ax = axes[0]
    ax.plot(concat_data, '.', label='Data', markersize=3, alpha=0.7)
    ax.set_ylabel('Data')
    ax.legend()

    ax = axes[1]
    ax.plot(concat_latent, label='Latent')
    ax.set_ylabel('Latent')
    ax.legend()

    ax = axes[2]
    ax.plot(concat_high_level_latent, label='High Level Latent')
    ax.set_ylabel('High Level Latent')
    ax.legend()

import imageio
class Logger:
    def __init__(self):
        self.training_batches = []
        self.training_losses = []
        self.training_losses_before_latent_optimization = []
        self.testing_batches = []
        self.testing_losses = []
        self.latent_values = []
        self.latent_gradients = []
        self.latent_updating_losses = [] # to store the loss at each optimization round
        self.latent_updating_latents = [] # to store the latent at each optimization round
        self.latent_updating_combined_inputs = []
        self.latent_updating_outputs = [] 
        self.predicted_outputs = []
        self.prediction_losses = []
        self.inputs = []
        self.llcids = []
        self.hlcids = []

    def log_updating_combined_input(self, combined_input):
        self.latent_updating_combined_inputs.append(combined_input)

    def log_updating_output(self, output):
        self.latent_updating_outputs.append(output)
        
    def log_updating_loss(self, loss):
        self.latent_updating_losses.append(loss)

    def log_training_batch(self, batch):
        self.training_batches.append(batch)

    def log_training_loss(self, loss):
        self.training_losses.append(loss)

    def log_testing_batch(self, batch):
        self.testing_batches.append(batch)

    def log_testing_loss(self, loss):
        self.testing_losses.append(loss)

    def log_latent_value(self, value):
        self.latent_values.append(value)

    def log_latent_gradient(self, gradient):
        self.latent_gradients.append(gradient)
    
    def log_training_loss_before_latent_optimization(self, loss):
        self.training_losses_before_latent_optimization.append(loss)

    def log_updating_latent(self, latent):
        self.latent_updating_latents.append(latent)

    def log_predicted_output(self, output):
        self.predicted_outputs.append(output)

    def log_prediction_loss(self, loss):
        self.prediction_losses.append(loss)
    
    def log_input(self, input):
        self.inputs.append(input)


def demo_latent_plots(model, inputs, gif_filename, criterion, LU_steps=100):
    demo_logger = Logger()
    model.init_latent()
    model.reset_latent(batch_size=inputs.shape[0], seq_len=inputs.shape[1])
    model.update_latent(inputs, criterion, logger=demo_logger, no_of_latent_steps=LU_steps)

    plt.ioff()
    plots = []
    example_batch_no = 0

    for optim_round in range((LU_steps)):
        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        ax = axes[0]
        im = demo_logger.latent_updating_latents[optim_round][example_batch_no, ...]
        ax.set_ylabel('Latent Dimension')
        ax.set_xlabel('Optimization Rounds')
        _ = ax.imshow(im.T, aspect='auto', cmap='viridis', interpolation='none')
        # write the min and max values of the latent in the title
        ax.set_title(f'Latent Optimization Round {optim_round}, Min: {im.min():.2f}, Max: {im.max():.2f}')

        ax = axes[1]
        lloss = [n[example_batch_no].mean() for n in demo_logger.latent_updating_losses]
        _ = ax.plot(lloss, linewidth=1, label='Loss')
        ax.scatter(optim_round, lloss[optim_round], c='r', s=10)
        ax.set_xlabel('Optimization Rounds')
        ax.set_ylabel('Loss')

        plt.savefig(f"./temp/latent_plot_{optim_round}.png")
        plots.append(f"./temp/latent_plot_{optim_round}.png")
        plt.close('all')

    images = [imageio.v2.imread(plot) for plot in plots]
    imageio.mimsave(f"{gif_filename}.gif", images, duration=0.03)
    plt.close('all')
    plt.ion()
#%%
def demo_latent_plots_with_outputs(model, inputs, gif_filename, criterion, config, LU_steps=100, example_batch_no = 0) :
    demo_logger = Logger()
    model.init_latent()
    model.reset_latent(batch_size=inputs.shape[0], seq_len=inputs.shape[1])
    model.update_latent(inputs, criterion, logger=demo_logger, no_of_latent_steps=LU_steps)

    plt.ioff()
    plots = []
    for optim_round in range((LU_steps)):
        fig, axes = plt.subplots(3, 1, figsize=(5, 4))
        ax = axes[0]
        ci = demo_logger.latent_updating_combined_inputs[optim_round][example_batch_no, ...]
        ii = ci[:, :-config.latent_dims[0]]
        if (ii.shape[-1] ) > 1: # if there are more than one features
            ax.imshow(ii.T, aspect='auto', cmap='viridis', interpolation='none')
            ax.set_ylabel('Feature')
        else: # if input is 1D
            ax.plot(ii, 'o', alpha = 1, markersize =1, linewidth=1, label = 'Input', color=obs_color)
            oi = demo_logger.latent_updating_outputs[optim_round][example_batch_no, ...]
            ax.plot(oi, 'o', alpha = 1, markersize =1, linewidth=1, label='Predicted Output', color=preds_color)
            ax.set_ylabel('Feature')
        ax.set_xticklabels([])
        ax.legend()
        # shade the background alternatively using ax span for each block
        ax.set_xlim(0, ii.shape[0])

        ax = axes[1]
        im = demo_logger.latent_updating_latents[optim_round][example_batch_no, ...]
        ax.set_ylabel('Latent')
        ax.set_xlabel('Time steps')
        _ = ax.imshow(im.T, aspect='auto', cmap='viridis', interpolation='none')
        # write the min and max values of the latent in the title
        axes[0].set_title(f'Latent Optimization Round {optim_round}, Min: {im.min():.2f}, Max: {im.max():.2f}')

        ax = axes[2]
        lloss = [n[example_batch_no].mean() for n in demo_logger.latent_updating_losses]
        _ = ax.plot(lloss, linewidth=1, label='Loss')
        ax.scatter(optim_round, lloss[optim_round], c='r', s=10)
        ax.set_xlabel('Optimization rounds')
        ax.set_ylabel('Loss')

        fig.tight_layout()
        plt.savefig(f"./temp/latent_plot_{optim_round}.png")
        plots.append(f"./temp/latent_plot_{optim_round}.png")
        plt.close('all')

    images = [imageio.v2.imread(plot) for plot in plots]
    imageio.mimsave(f"{gif_filename}.gif", images, duration=0.03)
    plt.close('all')
    plt.ion()

    # load and show GIF
    from IPython.display import Image
    Image(filename=gif_filename+'.gif')

#%%
obs_color = 'tab:grey'
preds_color = 'tab:red'

figure_dpi = 600


from torch.utils.data import Dataset, DataLoader   
class TaskDataset(Dataset):
    def __init__(self, no_of_blocks, config):
        self.num_blocks = no_of_blocks
        self.block_size = config.block_size
        self.latent_change_interval = config.latent_change_interval
        self.default_std = config.default_std
        self.high_level_latent_change_interval_in_blocks = config.high_level_latent_change_interval_in_blocks

        self.latent_values = [0.2, 0.5, 0.8]
        self.high_level_latent_values = [1, 2]
        self.rng = np.random.default_rng(0)
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.data_sequence = self.generate_data_sequence()
        self.config = config

    def __len__(self):
        return (self.num_blocks * self.block_size - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len
        data = self.data_sequence[start:end]
        latent = self.latent_sequence[start:end]
        high_level_latent = self.high_level_latent_sequence[start:end]
        data = torch.tensor(data, dtype=torch.float32).reshape( -1, 1)
        latent = torch.tensor(latent, dtype=torch.float32).reshape( -1, 1)    
        high_level_latent = torch.tensor(high_level_latent, dtype=torch.float32).reshape(-1, 1)

        return data, latent, high_level_latent

    def generate_latent_sequence(self):
        latent_sequence = []
        for i in range(self.num_blocks):
            if i % self.latent_change_interval == 0:
                latent = self.rng.choice(self.latent_values)
            latent_sequence.extend([latent] * self.block_size)
        return latent_sequence

    def generate_high_level_latent_sequence(self):
        high_level_latent_sequence = []
        for i in range(self.num_blocks):
            if i % self.high_level_latent_change_interval_in_blocks == 0:
                high_level_latent = self.rng.choice(self.high_level_latent_values)
            high_level_latent_sequence.extend([high_level_latent] * self.block_size)
        return high_level_latent_sequence

    def generate_data_sequence(self):
        data_sequence = []
        for i in range(self.num_blocks):
            block_idx = i * self.block_size
            mean = self.latent_sequence[block_idx]
            std = self.default_std
            seed = self.high_level_latent_sequence[block_idx]
            self.data_rng = np.random.default_rng(seed)
            block_data = self.data_rng.normal(mean, std, self.block_size)
            data_sequence.extend(block_data)

        return data_sequence

# %%
