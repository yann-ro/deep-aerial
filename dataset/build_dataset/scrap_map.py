import os


def download_tile():
    x_min = 2.33
    x_max = 2.33115
    y_min = 48.87
    y_max = 48.87075
    source = "bing.xml"
    output = "im.tiff"
    output_res = 0.5

    os.system(
        f"gdal_translate -projwin {x_min} {y_max} {x_max} {y_min} -projwin_srs EPSG:4326 -tr {output_res} {output_res} -r bilinear -co COMPRESS=DEFLATE {source} {output}"
    )


if __name__ == "__main__":
    download_tile()
