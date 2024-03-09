import numpy as np
import matplotlib.pyplot as plt


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def main(root_dir, num_images):

    angle_err = np.zeros(num_images)
    print()
    for i_image in range(num_images):
        print(f"\rImage {i_image}", end="")

        vpt_ground_truth = np.loadtxt(f"{root_dir}/vpoint_{i_image}.txt")[:-1]
        vpt_ground_truth /= np.linalg.norm(vpt_ground_truth)

        vpt_min_enc_tri = np.loadtxt(f"{root_dir}/tri_vpoint_{i_image}.txt")[:-1]
        vpt_min_enc_tri /= np.linalg.norm(vpt_min_enc_tri)

        angle_err[i_image] = (
            np.arccos(np.abs(vpt_ground_truth.dot(vpt_min_enc_tri)).clip(max=1))
            * 180
            / np.pi
        )

    sorted_angle_err = np.sort(np.asarray(angle_err))
    y = (1 + np.arange(num_images)) / num_images

    plt.plot(sorted_angle_err, y, label="Min. enc. triangle")
    print(
        " | ".join(
            [f"{AA(sorted_angle_err, y, th):.3f}" for th in [0.5, 1, 2, 5, 10, 20]]
        )
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel("Angle error (deg)")
    plt.ylabel("Percentile rank")
    plt.savefig(f"aa_cdf_{root_dir}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    # main(root_dir="tri_vpoint_no_blob", num_images=20000)
    main(root_dir="tri_vpoint_with_blob", num_images=20000)
