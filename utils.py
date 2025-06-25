import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# ----------------------------
def img_data(imgPath, disp=False):
    orig_img = Image.open(imgPath)
    img_size_kb = os.stat(imgPath).st_size / 1024
    ori_pixels = np.array(orig_img.getdata()).reshape(*orig_img.size, -1)
    img_dim = ori_pixels.shape
    if disp:
        plt.imshow(orig_img)
        plt.show()
    return {
        'img_size_kb': round(img_size_kb, 2),
        'img_dim': img_dim
    }

# ----------------------------
def pca_compose(imgPath):
    orig_img = Image.open(imgPath)
    img = np.array(orig_img.getdata()).reshape(*orig_img.size, -1)
    img_t = np.transpose(img)
    pca_channel = {}

    for i in range(img.shape[-1]):
        channel = img_t[i].reshape(*img.shape[:-1])
        pca = PCA(random_state=42)
        fit_pca = pca.fit_transform(channel)
        pca_channel[i] = (pca, fit_pca)

    return pca_channel

# ----------------------------
def explained_var_n(pca_channel, n):
    var_exp_channel = []

    for channel in pca_channel:
        pca, _ = pca_channel[channel]
        var_exp_channel.append(sum(pca.explained_variance_ratio_[:n]))

    return sum(var_exp_channel) / len(var_exp_channel)

# ----------------------------
def variance_added_pc(pca_channel, save_path):
    var_exp_channel = []
    for channel in pca_channel:
        pca, _ = pca_channel[channel]
        var_exp_channel.append(pca.explained_variance_ratio_)

    var_exp = (var_exp_channel[0] + var_exp_channel[1] + var_exp_channel[2]) / 3
    x = list(var_exp)
    y = list(range(1, len(x) + 1))

    plt.figure()
    plt.yticks(np.arange(0, max(x) + 0.05, 0.05))
    plt.xticks(np.arange(1, 21, 1))
    plt.title("Individual Variance for each Principal Component")
    plt.ylabel('Variance')
    plt.xlabel('Principal Component')
    plt.bar(y[:20], x[:20], color='black')
    plt.tight_layout()
    out_path = os.path.join(save_path, "variance_bar.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

# ----------------------------
def plot_variance_pc(pca_channel, save_path):
    pca, fit_pca = pca_channel[0]
    exp_var = {}

    for i in range(len(pca.components_)):
        var_exp = explained_var_n(pca_channel, i)
        exp_var[i + 1] = var_exp

    lists = sorted(exp_var.items())
    x, y = zip(*lists)

    pt90 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.9)
    pt95 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.95)

    plt.figure()
    plt.plot(x, y, label="Variance Explained")
    plt.vlines(x=x[pt90], ymin=0, ymax=y[pt90], colors='green', ls=':', lw=2,
               label=f'90% Variance Explained: n = {x[pt90]}')
    plt.vlines(x=x[pt95], ymin=0, ymax=y[pt95], colors='red', ls=':', lw=2,
               label=f'95% Variance Explained: n = {x[pt95]}')

    plt.xticks(np.arange(min(x), max(x), 100))
    plt.yticks(np.arange(0, max(y), 0.1))
    plt.legend(loc="lower right")
    plt.title("Variance vs Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Variance Explained")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_path, "variance_line.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

# ----------------------------
def pca_transform(pca_channel, n_components):
    temp_res = []
    for channel in range(len(pca_channel)):
        pca, fit_pca = pca_channel[channel]
        pca_pixel = fit_pca[:, :n_components]
        pca_comp = pca.components_[:n_components, :]
        compressed_pixel = np.dot(pca_pixel, pca_comp) + pca.mean_
        temp_res.append(compressed_pixel)

    compressed_image = np.transpose(temp_res)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    return compressed_image

# ----------------------------
def full_pca_process(img_path, output_dir, plot_dir, n_components=1200):
    # Step 1: Original stats
    data_dict_ori = img_data(img_path)
    orig_size = data_dict_ori['img_size_kb']
    orig_shape = data_dict_ori['img_dim']

    # Step 2: PCA Decomposition
    pca_channel = pca_compose(img_path)

    # Step 3: PCA Compression
    compressed_image = pca_transform(pca_channel, n_components=n_components)
    compressed_img_path = os.path.join(output_dir, "compressed_img.jpeg")
    Image.fromarray(compressed_image).save(compressed_img_path)

    # Step 4: Compressed stats
    data_dict_comp = img_data(compressed_img_path)
    comp_size = data_dict_comp['img_size_kb']
    comp_shape = data_dict_comp['img_dim']
    compression_percent = round(100 - (comp_size / orig_size) * 100, 2)

    # Step 5: Plots
    bar_plot_path = variance_added_pc(pca_channel, plot_dir)
    line_plot_path = plot_variance_pc(pca_channel, plot_dir)

    return {
        'orig_path': img_path,
        'orig_size': orig_size,
        'orig_shape': orig_shape,
        'comp_path': compressed_img_path,
        'comp_size': comp_size,
        'comp_shape': comp_shape,
        'compression_percent': compression_percent,
        'bar_plot': bar_plot_path,
        'line_plot': line_plot_path
    }
