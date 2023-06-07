import matplotlib.pyplot as plt
import numpy as np
import math
import os

eval_env_type = ['normal', 'color_hard', 'video_easy', 'video_hard']


def average_over_several_runs(folder, action_repeat):
    mean_all = []
    std_all = []
    for env_type in range(len(eval_env_type)):
        data_all = []
        min_length = np.inf
        runs = os.listdir(folder)
        for i in range(len(runs)):
            data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
            evaluation_freq = (data[2, -2]-data[1, -2])*action_repeat
            data_all.append(data[:, 2+env_type])
            if data.shape[0] < min_length:
                min_length = data.shape[0]
        average = np.zeros([len(runs), min_length])
        for i in range(len(runs)):
            average[i, :] = data_all[i][:min_length]
        mean = np.mean(average, axis=0)
        mean_all.append(mean)
        std = np.std(average, axis=0)
        std_all.append(std)

    return mean_all, std_all, evaluation_freq/1000


def plot_several_folders(prefix, folders, action_repeat, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (9, 9)
    fig, axs = plt.subplots(2, 2)
    for i in range(len(folders)):
        folder_name = 'logs/saved_logs/'+prefix+folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name, action_repeat)
        for j in range(len(eval_env_type)):
            # plot variance
            axs[int(j/2)][j-2*(int(j/2))].fill_between(eval_freq*range(len(mean_all[j])),
                    mean_all[j] - std_all[j]/math.sqrt(num_runs),
                    mean_all[j] + std_all[j]/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                axs[int(j/2)][j-2*(int(j/2))].plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=label_list[i])
            else:
                axs[int(j/2)][j-2*(int(j/2))].plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=folders[i])

            axs[int(j/2)][j-2*(int(j/2))].set_xlabel('env steps/k')
            axs[int(j/2)][j-2*(int(j/2))].set_ylabel('episode reward')
            axs[int(j/2)][j-2*(int(j/2))].legend(fontsize=8)
            axs[int(j/2)][j-2*(int(j/2))].set_title(eval_env_type[j])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('logs/saved_fig/'+title)


prefix = 'walker_walk/'
action_repeat = 4
folders_1 = ['svea_random_overlay', 'svea_random_alpha', 'cut_random_overlay', 'resize_mix', 'cut_mix']
folders_2 = ['svea_random_conv', 'random_conv_gaussian_not_detach_encoder', 'random_conv_gaussian_detach_encoder',
             'random_conv_gaussian_without_tanh_detach_encoder', 'random_conv_beta_detach_encoder',
             'random_conv_categorical_detach_encoder']
folders_3 = ['svea_random_overlay', 'svea_random_overlay_q_diff', 'svea_random_overlay_q_diff_div_image_diff',
             'svea_random_overlay_q_diff_critic_proj_grad', 'cut_random_overlay',
             'svea_cut_random_overlay_q_diff_critic_proj_grad',
             'svea_cut_random_overlay_q_diff_AC_proj_grad',
             'svea_cut_random_overlay_q_diff_kl_diff_AC_proj_grad']
plot_several_folders(prefix, folders_1, action_repeat, title='walker_walk_more_da')
plot_several_folders(prefix, folders_2, action_repeat, title='walker_walk_distributional_random_conv')
plot_several_folders(prefix, folders_3, action_repeat, title='walker_walk_weighted_random_overlay')


prefix = 'ball_in_cup_catch/'
action_repeat = 4
folders_1 = ['svea_random_overlay', 'svea_cut_random_overlay_q_diff_kl_diff_AC_proj_grad',
             'svea_cut_random_overlay_2imgs_q_diff_kl_diff_AC_proj_grad']
plot_several_folders(prefix, folders_1, action_repeat, title='ball_in_cup_catch_weighted_random_overlay')

prefix = 'cartpole_swingup/'
action_repeat = 8
folders_1 = ['svea_random_overlay', 'svea_cut_random_overlay_q_diff_kl_diff_AC_proj_grad']
plot_several_folders(prefix, folders_1, action_repeat, title='cartpole_swingup_weighted_random_overlay')
