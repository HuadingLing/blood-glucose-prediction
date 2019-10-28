import matplotlib.pyplot as plt
import numpy as np

def plot_with_std(y_ture, y_pred_mean, y_pred_std, coeffi = 3, title=" ", save_file_name = " "):
    delta = coeffi * np.abs(y_pred_std)
    y_pred_upper, y_pred_lower = y_pred_mean + delta, y_pred_mean - delta
    #x = np.linspace(0, len(y_ture))
    plt.plot(y_ture, 'g')
    plt.plot(y_pred_mean, 'b')
    plt.plot(y_pred_upper, 'r')
    plt.plot(y_pred_lower, 'r')
    plt.ylim(0, 400)
    plt.xlabel('time')
    plt.ylabel('glucose level')
    plt.title(title)
    if save_file_name != " ":
        plt.savefig(save_file_name)
    plt.show()
    
def plot_without_std(y_ture, y_pred, title=" ", save_file_name = " "):
    plt.plot(y_ture, 'g')
    plt.plot(y_pred, 'b')
    plt.ylim(0, 400)
    plt.xlabel('time')
    plt.ylabel('glucose level')
    plt.title(title)
    if save_file_name != " ":
        plt.savefig(save_file_name)
    plt.show()