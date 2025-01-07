import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Title halaman EDA
def run():
    st.title("Exploratory Data Analysis (EDA) - Waste Image Classification")

    # Bagian 1: Visualisasi Distribusi Kelas
    st.header("1. Class Distribution Visualization")

    # Simulasi daftar label
    labels = ['Cardboard', 'Plastic', 'Paper', 'Plastic', 'Cardboard', 'Paper', 'Plastic', 'Cardboard']

    # Fungsi untuk membuat bar chart
    def plot_class_distribution_from_folders(labels, title="Image Distribution per Class", color='skyblue'):
        """
        Generates a bar chart showing the distribution of images per class
        using data from a list of labels.

        Args:
            labels (list): List of class labels corresponding to images.
            title (str): Title of the chart.
            color (str): Color of the bars.
        """
        # Count the number of images per class
        label_counts = Counter(labels)

        # Sort classes alphabetically (or modify as needed)
        sorted_labels = sorted(label_counts.items())

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            [label for label, _ in sorted_labels],  # Class names
            [count for _, count in sorted_labels],  # Counts
            color=color
        )

        # Add count labels above each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, str(yval), ha='center', fontsize=10, color='black')

        # Add titles and axis labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Classes", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Return the figure
        return fig

    # Tampilkan grafik distribusi kelas
    fig = plot_class_distribution_from_folders(labels)
    st.pyplot(fig)