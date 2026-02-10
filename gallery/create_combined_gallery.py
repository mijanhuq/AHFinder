"""
Create a combined gallery image showing all horizons in a grid.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def create_combined_gallery():
    gallery_dir = os.path.dirname(os.path.abspath(__file__))

    # Get all horizon images
    images = sorted(glob.glob(os.path.join(gallery_dir, "horizon_*.png")))

    # Parse parameters from filenames
    parsed = []
    for img_path in images:
        filename = os.path.basename(img_path)
        # horizon_a0p25_v0p3.png
        parts = filename.replace("horizon_", "").replace(".png", "").split("_")
        a = float(parts[0].replace("a", "").replace("p", "."))
        v = float(parts[1].replace("v", "").replace("p", "."))
        parsed.append({'a': a, 'v': v, 'path': img_path})

    # Get unique velocities and spins
    velocities = sorted(set(p['v'] for p in parsed))
    spins = sorted(set(p['a'] for p in parsed))

    n_rows = len(velocities)
    n_cols = len(spins)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

    # Add row and column labels
    for row, v in enumerate(velocities):
        for col, a in enumerate(spins):
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # Find matching image
            matching = [p for p in parsed if p['a'] == a and p['v'] == v]

            if matching:
                img = mpimg.imread(matching[0]['path'])
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Failed\nto converge", ha='center', va='center',
                       fontsize=12, transform=ax.transAxes)
                ax.set_facecolor('#ffdddd')
                ax.axis('off')

            # Add title for top row
            if row == 0:
                if a == 0:
                    ax.set_title(f"Schwarzschild", fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f"a = {a}", fontsize=12, fontweight='bold')

        # Add row label
        ax = axes[row, 0] if n_rows > 1 else axes[0]
        ax.text(-0.15, 0.5, f"v = {v}c", rotation=90, ha='center', va='center',
               fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.suptitle("Boosted Kerr Black Hole Horizons\n(Lorentz contraction visible at higher velocities)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(gallery_dir, "gallery_combined.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()

if __name__ == "__main__":
    create_combined_gallery()
