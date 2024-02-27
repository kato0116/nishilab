import matplotlib.pyplot as plt
def show_images(images, labels, epoch, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_xlabel(labels[i].item())
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.savefig(f"/root/log/epoch_{epoch}.png")