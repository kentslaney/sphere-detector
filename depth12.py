import pathlib, sys
import tensorflow as tf
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from static import Depth
sys.path.pop(0)

dsfs = local / "assets" / "datasets"
balls_depth_records = dsfs / "depth_balls.tfrecord"
ballnt_depth_records = dsfs / "depth_ballnt.tfrecord"

def read_balls_depth_records():
    depth_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([17 * 4], tf.int64)
    }

    ds = tf.data.TFRecordDataset(str(balls_depth_records))

    def _parse_depth_function(x):
        res = tf.io.parse_single_example(x, depth_feature_description)
        return {
            'file_name': res['file_name'],
            'image': tf.io.parse_tensor(res['image'], out_type=tf.uint8),
            'depth': tf.io.parse_tensor(res['depth'], out_type=tf.float32),
            'label': tf.io.parse_tensor(res['label'], out_type=tf.int64),
            'bbox': tf.reshape(tf.cast(res['bbox'], tf.int32), (17, 4))
        }

    return ds.map(_parse_depth_function)

def read_ballnt_depth_records():
    ballnt_depth_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

    ds = tf.data.TFRecordDataset(str(ballnt_depth_records))

    def _parse_ballnt_depth_function(x):
        res = tf.io.parse_single_example(x, ballnt_depth_feature_description)
        return {
            'file_name': res['file_name'],
            'image': tf.io.parse_tensor(res['image'], out_type=tf.uint8),
            'depth': tf.io.parse_tensor(res['depth'], out_type=tf.float32),
            'label': tf.io.parse_tensor(res['label'], out_type=tf.int64),
        }

    return ds.map(_parse_ballnt_depth_function)

if __name__ == "__main__":
    from scipy.stats import expon

    cache_dir = local / "cache"
    analysis_file = cache_dir / "density_analysis.npy"

    if analysis_file.exists():
        print(f"Loading cached results from {analysis_file}...")
        results = np.load(analysis_file, allow_pickle=True).item()
        bins = results['bins']
        total_inside_hist = results['total_inside_hist']
        total_outside_hist = results['total_outside_hist']
        mean_inside = results['mean_inside']
        mean_outside = results['mean_outside']
        inside_count = results['inside_count']
        outside_count = results['outside_count']
    else:
        print("No cached analysis file found. Processing datasets...")
        ballnt_ds = read_ballnt_depth_records()
        balls_ds = read_balls_depth_records()

        # Define a common set of bins for all histograms
        num_bins = 200
        # Exponentially spaced bins from 0.01 to 100
        bins = np.logspace(-2, 2, num_bins + 1)

        total_inside_hist = np.zeros(num_bins, dtype=np.int64)
        total_outside_hist = np.zeros(num_bins, dtype=np.int64)

        inside_sum = 0.0
        inside_count = 0
        outside_sum = 0.0
        outside_count = 0

        # It's good practice to show progress for long-running tasks
        from tqdm import tqdm
        for example in tqdm(balls_ds, desc="Sampling 'inside' densities", total=10382):
            depth_map = jnp.array(example['depth'].numpy())
            bboxes = example['bbox'].numpy()

            density_map = np.array(Depth(depth_map).density())

            mask = np.zeros(density_map.shape, dtype=bool)
            if bboxes.ndim == 2: # Ensure bboxes is not empty
                for xmin, ymin, xmax, ymax in bboxes:
                    if xmin != -1:  # Bounding boxes are padded with -1
                        mask[ymin:ymax, xmin:xmax] = True

            inside_values = density_map[mask]
            inside_values = inside_values[inside_values > 0]  # Exclude zeros

            # Accumulate sums and counts for mean calculation
            inside_sum += np.sum(inside_values)
            inside_count += len(inside_values)

            total_inside_hist += np.histogram(inside_values, bins=bins)[0]

        for example in tqdm(ballnt_ds, desc="Sampling 'outside' densities", total=10382):
            depth_map = jnp.array(example['depth'].numpy())
            density_map = np.array(Depth(depth_map).density())
            outside_values = density_map[density_map > 0].flatten() # Exclude zeros

            outside_sum += np.sum(outside_values)
            outside_count += len(outside_values)

            total_outside_hist += np.histogram(outside_values, bins=bins)[0]

        # Calculate means
        mean_inside = inside_sum / inside_count if inside_count > 0 else 0
        mean_outside = outside_sum / outside_count if outside_count > 0 else 0

        # Save the results to a .npy file
        cache_dir.mkdir(exist_ok=True)
        np.save(analysis_file, {
            'bins': bins,

            'total_outside_hist': total_outside_hist,
            'mean_outside': mean_outside,
            'outside_count': outside_count,

            'total_inside_hist': total_inside_hist,
            'mean_inside': mean_inside,
            'inside_count': inside_count,
        })

    # Normalize histograms to get probability mass function
    inside_pmf = total_inside_hist / (total_inside_hist.sum() * np.diff(bins))
    outside_pmf = total_outside_hist / (total_outside_hist.sum() * np.diff(bins))

    # Fit exponential distributions
    # For an exponential distribution, the scale parameter (1/lambda) is the mean.
    # The location parameter is assumed to be 0.
    inside_fit = expon(scale=mean_inside)
    outside_fit = expon(scale=mean_outside)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    ratio = inside_count / (inside_count + outside_count)

    boundary = np.log(inside_count / mean_inside) - np.log(outside_count / mean_outside)
    boundary /= (1 / mean_inside) - (1 / mean_outside)

    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, inside_pmf, width=np.diff(bins), alpha=0.6, label=f'Cropped detection examples ({ratio:.2%})')
    plt.plot(bin_centers, inside_fit.pdf(bin_centers), 'b-', lw=2, label=f'postive exp fit (mean={mean_inside:.2f})')

    plt.gca().set_xscale('log')

    plt.bar(bin_centers, outside_pmf, width=np.diff(bins), alpha=0.6, label=f'Control images from ImageNet-1k ({1 - ratio:.2%})')
    plt.plot(bin_centers, outside_fit.pdf(bin_centers), 'r-', lw=2, label=f'negative exp fit (mean={mean_outside:.2f})')

    plt.axvline(boundary, color='k', linestyle='--', label='Bayes classification threshold')

    # Add a custom tick for the boundary
    ax = plt.gca()
    ax.set_xticks(list(ax.get_xticks()) + [boundary])
    ax.text(boundary, -0.05, "≈ 3",
            transform=ax.get_xaxis_transform(),
            ha='center', va='top')

    ax.set_xlim(1e-2, 1e2)

    plt.title('Center (Non-Zero) Density Distribution and Exponential Fit')
    plt.xlabel('Density Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()