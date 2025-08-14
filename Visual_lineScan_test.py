"""
Visual Linescan Parameter Tuning Tool
=====================================

This script helps you determine optimal peak detection parameters for your analysis.
It shows individual ROI traces with detected peaks overlaid, allowing you to:

1. See which peaks are being detected
2. Adjust parameters to catch real peaks while avoiding noise
3. Get the parameter values to use in master_pipeline.py

INSTRUCTIONS:
1. The script will automatically select a random CSV file from your configured directory
2. Choose which channel to analyze for peaks
3. Adjust the peak detection parameters until the results look good
4. Copy the working parameters to master_pipeline.py
"""

import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys

# =============================================================================
# USER CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# Use the same configuration as master_pipeline.py
EXPERIMENT_DIR = Path('mitochondrial_analysis')
CSV_SUBFOLDER = 'csv_files'  # Set to None if CSVs are directly in EXPERIMENT_DIR

# Which channel to analyze for peaks (the protein channel you're interested in)
PEAK_CHANNEL = 'Channel_4'  # Options: 'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4'

# Which channels to display in the plot (for visual comparison)
CHANNELS_TO_DISPLAY = ['Channel_4']  # Show these channels in the plot, Options: 'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4'

# Peak detection parameters to test
# Adjust these values until peak detection looks good
PEAK_PARAMETERS = {
    'prominence': 1000,  # Minimum peak prominence (how much peak stands out)
    'width': 4,  # Minimum peak width in pixels
    'height': 5000,  # Minimum peak height (intensity value)
    'distance': 6  # Minimum distance between peaks (pixels)
}

# Display settings
MAX_ROIS_TO_SHOW = None  # How many ROIs to display (set to None for all)
SHOW_PEAK_INFO = True  # Print peak statistics to console
USE_SPECIFIC_FILE = None  # Set to a filename to use a specific file instead of random


# =============================================================================
# ANALYSIS CODE - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def get_all_csv_files():
    """Get all CSV files from the configured directory"""
    if CSV_SUBFOLDER:
        csv_path = EXPERIMENT_DIR / CSV_SUBFOLDER
    else:
        csv_path = EXPERIMENT_DIR

    if not csv_path.exists():
        print(f"Error: CSV directory not found: {csv_path}")
        print("Please check your EXPERIMENT_DIR and CSV_SUBFOLDER settings.")
        sys.exit(1)

    csv_files = list(csv_path.glob('*.csv'))

    if not csv_files:
        print(f"Error: No CSV files found in {csv_path}")
        sys.exit(1)

    return csv_files


def plot_roi_with_peaks(roi_data, title, peak_results):
    """Plot a single ROI with detected peaks highlighted"""
    plt.figure(figsize=(12, 6))

    # Plot channel data
    colors = {'Channel_1': 'red', 'Channel_2': 'green', 'Channel_3': 'blue', 'Channel_4': 'orange'}

    for channel in CHANNELS_TO_DISPLAY:
        if channel in roi_data.columns:
            plt.plot(roi_data[channel], color=colors.get(channel, 'black'),
                     linewidth=2, label=channel, alpha=0.8)

    # Highlight detected peaks
    if len(peak_results[0]) > 0:
        peak_positions = peak_results[0]
        peak_properties = peak_results[1]

        for i, peak_pos in enumerate(peak_positions):
            # Vertical line at peak position
            plt.axvline(peak_pos, color='black', linestyle='--', alpha=0.7,
                        label='Peak' if i == 0 else "")

            # Highlight peak width
            if 'left_ips' in peak_properties and 'right_ips' in peak_properties:
                left_edge = peak_properties['left_ips'][i]
                right_edge = peak_properties['right_ips'][i]
                plt.axvspan(left_edge, right_edge, alpha=0.2, color='gray')

            # Add peak number annotation
            peak_height = roi_data[PEAK_CHANNEL].iloc[peak_pos]
            plt.annotate(f'P{i + 1}', xy=(peak_pos, peak_height),
                         xytext=(peak_pos, peak_height + peak_height * 0.1),
                         ha='center', fontsize=10, fontweight='bold')

    plt.title(f'{title}\n'
              f'Parameters: prominence={PEAK_PARAMETERS["prominence"]}, '
              f'width={PEAK_PARAMETERS["width"]}, height={PEAK_PARAMETERS["height"]}')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_peak_statistics(peak_results, roi_identifier):
    """Print detailed statistics about detected peaks"""
    if len(peak_results[0]) == 0:
        print(f"{roi_identifier}: No peaks detected")
        return

    peak_properties = peak_results[1]
    print(f"\n{roi_identifier}: {len(peak_results[0])} peaks detected")
    print("-" * 40)

    for i, peak_pos in enumerate(peak_results[0]):
        print(f"Peak {i + 1}:")
        print(f"  Position: {peak_pos} pixels")
        print(f"  Height: {peak_properties['peak_heights'][i]:.0f}")
        print(f"  Width: {peak_properties['widths'][i]:.1f} pixels")
        print(f"  Prominence: {peak_properties['prominences'][i]:.0f}")


def main():
    """Main analysis function"""
    print("Visual Linescan Parameter Tuning Tool")
    print("=" * 50)
    print(f"Peak detection channel: {PEAK_CHANNEL}")
    print(f"Current parameters: {PEAK_PARAMETERS}")

    # Get all available CSV files
    all_csv_files = get_all_csv_files()
    print(f"\nFound {len(all_csv_files)} CSV files in directory")

    # Track which ROIs we've already shown
    shown_rois = set()  # Will store tuples of (filename, roi_index)

    # First, count total available ROIs
    print("Counting available ROIs...")
    total_available_rois = 0
    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            num_rois = len([col for col in df.columns if 'distance' in col])
            total_available_rois += num_rois
        except:
            continue

    print(f"Total ROIs available across all files: {total_available_rois}")

    # Determine how many plots to show
    if MAX_ROIS_TO_SHOW:
        plots_to_show = min(MAX_ROIS_TO_SHOW, total_available_rois)
        print(f"Will show up to {plots_to_show} plots (limited by MAX_ROIS_TO_SHOW)")
    else:
        plots_to_show = total_available_rois
        print(f"Will continue until all {total_available_rois} ROIs have been shown")

    print("\n")

    total_peaks = 0
    plots_shown = 0
    consecutive_failures = 0
    max_consecutive_failures = 50  # Stop if we can't find new ROIs after many attempts

    while plots_shown < plots_to_show:
        # Select a random CSV file
        csv_file = random.choice(all_csv_files)

        # Load the CSV
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("\nUnable to find more valid ROIs after many attempts.")
                break
            continue

        # Get all valid ROIs from this file that we haven't shown yet
        num_rois = len([col for col in df.columns if 'distance' in col])
        valid_rois = []

        for roi_idx in range(num_rois):
            # Skip if we've already shown this specific ROI
            if (csv_file.name, roi_idx) in shown_rois:
                continue

            # Quick check if ROI has valid data
            try:
                distance_col = df[f'ROI_{roi_idx} distance (micron)']
                channel_col = f'ROI_{roi_idx} {PEAK_CHANNEL}'

                # Find valid length
                valid_length = len(distance_col)
                for j in range(1, len(distance_col)):
                    if distance_col.iloc[j] == 0:
                        valid_length = j
                        break

                # Check if ROI is valid
                if valid_length >= 10 and channel_col in df.columns:
                    valid_rois.append(roi_idx)
            except:
                continue

        if not valid_rois:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("\nAll valid ROIs have been shown!")
                break
            continue  # Try another file

        # Reset failure counter since we found valid ROIs
        consecutive_failures = 0

        # Randomly select one of the valid ROIs we haven't shown yet
        selected_roi_idx = random.choice(valid_rois)

        # Mark this ROI as shown
        shown_rois.add((csv_file.name, selected_roi_idx))

        print(f"\n--- Plot {plots_shown + 1} (ROI {len(shown_rois)}/{total_available_rois}) ---")
        print(f"File: {csv_file.name}")
        print(f"ROI: {selected_roi_idx}")

        # Extract the selected ROI data
        roi_data = pd.DataFrame()
        roi_data['Distance'] = df[f'ROI_{selected_roi_idx} distance (micron)']

        # Find where distance becomes 0 (end of valid data)
        valid_length = len(roi_data)
        for j in range(1, len(roi_data)):
            if roi_data['Distance'].iloc[j] == 0:
                valid_length = j
                break

        # Extract all channel data and trim to valid length
        for ch_num in range(1, 5):
            col_name = f'ROI_{selected_roi_idx} Channel_{ch_num}'
            if col_name in df.columns:
                roi_data[f'Channel_{ch_num}'] = df[col_name]

        roi_data = roi_data.iloc[:valid_length]

        # Detect peaks
        peak_results = find_peaks(
            roi_data[PEAK_CHANNEL],
            prominence=PEAK_PARAMETERS['prominence'],
            width=PEAK_PARAMETERS['width'],
            height=PEAK_PARAMETERS['height'],
            distance=PEAK_PARAMETERS['distance']
        )

        total_peaks += len(peak_results[0])
        plots_shown += 1

        # Display results
        if SHOW_PEAK_INFO:
            print_peak_statistics(peak_results, f"{csv_file.name} ROI_{selected_roi_idx}")

        # Plot with filename and ROI in title
        plot_roi_with_peaks(roi_data, f"{csv_file.name} ROI_{selected_roi_idx}", peak_results)

    # Completion message
    if len(shown_rois) >= total_available_rois:
        print(f"\n{'=' * 50}")
        print("ALL ROIs HAVE BEEN VIEWED!")
        print(f"{'=' * 50}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"SUMMARY:")
    print(f"Plots shown: {plots_shown}")
    print(f"Unique ROIs viewed: {len(shown_rois)} out of {total_available_rois} total")
    print(f"Total peaks detected: {total_peaks}")
    if plots_shown > 0:
        print(f"Average peaks per ROI: {total_peaks / plots_shown:.1f}")
    print(f"\nCurrent parameters:")
    print(f"  Width: {PEAK_PARAMETERS['width']}")
    print(f"  Height: {PEAK_PARAMETERS['height']}")
    print(f"  Prominence: {PEAK_PARAMETERS['prominence']}")
    print(f"  Distance: {PEAK_PARAMETERS['distance']}")
    print(f"\nCopy this to master_pipeline.py for {PEAK_CHANNEL} analysis:")
    channel_num = int(PEAK_CHANNEL.split('_')[1])
    print(f"{channel_num}: [{PEAK_PARAMETERS['width']}, {PEAK_PARAMETERS['height']}, {PEAK_PARAMETERS['prominence']}, {PEAK_PARAMETERS['distance']}],")


if __name__ == "__main__":
    main()