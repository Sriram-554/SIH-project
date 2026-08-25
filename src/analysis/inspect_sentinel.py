from pathlib import Path

DATA_DIR = Path("data")

safe_folders = list(DATA_DIR.glob("*.SAFE"))

if not safe_folders:
    print("[!] No Sentinel-2 SAFE folder found.")
    raise SystemExit

safe = safe_folders[0]

print("=" * 60)
print("SATQUERY - SENTINEL-2 DATA INSPECTOR")
print("=" * 60)

print(f"\nSAFE PRODUCT:")
print(safe.name)

print("\nSearching for image files...\n")

image_files = []

for path in safe.rglob("*"):
    if path.is_file() and path.suffix.lower() in [".jp2", ".tif", ".tiff"]:
        image_files.append(path)

if not image_files:
    print("[!] No JP2/TIFF image files found.")
else:
    print(f"Found {len(image_files)} image files:\n")

    for image in image_files:
        print(image.relative_to(safe))

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)