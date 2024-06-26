#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2024-05-31
Purpose: RGB Panicle Detection
"""

import argparse
import os
import sys

from multiprocessing import process
from detecto import core, utils, visualize
import glob
import shutil
import cv2
from detecto.core import Model
import numpy as np
import tifffile as tifi
from osgeo import gdal
import pyproj
import utm
import json
import pandas as pd
import multiprocessing
import warnings
from pyproj import Proj
warnings.filterwarnings("ignore")


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='RGB Panicle Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='img_list',
                        help='Directory containing files')

    parser.add_argument('-m',
                        '--model',
                        help='A .pth model file',
                        metavar='model',
                        type=str,
                        default=None,
                        required=True)
    
    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='detect_out')
    
    parser.add_argument('-c',
                        '--detect_class',
                        nargs='+',
                        help='Classes to detect',
                        metavar='detect_class',
                        type=str,
                        required=True)
    
    parser.add_argument('-d',
                        '--date',
                        help='Scan date',
                        metavar='date',
                        type=str,
                        default=None,
                        required=True)
    
    parser.add_argument('-g',
                        '--geojson',
                        help='GeoJSON containing plot boundaries',
                        metavar='str',
                        type=str,
                        default=None,
                        required=True)
    
    parser.add_argument('-t',
                        '--type',
                        help='Specify if FLIR or RGB images',
                        default='RGB',
                        choices=['FLIR', 'RGB', 'DRONE'])
    
    return parser.parse_args()


# --------------------------------------------------
def get_min_max(box):
    min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    return min_x, min_y, max_x, max_y


# --------------------------------------------------
def get_genotype(plot, geojson):
    with open(geojson) as f:
        data = json.load(f)

    for feat in data['features']:
        if feat.get('properties')['ID']==plot:
            genotype = feat.get('properties').get('genotype')

    return genotype


# --------------------------------------------------
def pixel2geocoord(one_img, x_pix, y_pix):
    ds = gdal.Open(one_img)
    c, a, b, f, d, e = ds.GetGeoTransform()
    lon = a * int(x_pix) + b * int(y_pix) + a * 0.5 + b * 0.5 + c
    lat = d * int(x_pix) + e * int(y_pix) + d * 0.5 + e * 0.5 + f

    return (lat, lon)


# --------------------------------------------------
def open_image(img_path):

    args = get_args()

    if args.type == 'FLIR':
        a_img = tifi.imread(img_path)
        a_img = cv2.cvtColor(a_img, cv2.COLOR_GRAY2BGR)
        a_img = cv2.normalize(a_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    else:
        a_img = tifi.imread(img_path)
        a_img = np.array(a_img)
        
        if a_img.shape[2]==4:
            a_img = np.array(a_img)[:,:,:3]

    return a_img


# --------------------------------------------------
def process_image(img):

    args = get_args()
    cont_cnt = 0
    lett_dict = {}
    model = core.Model.load(args.model, args.detect_class)

    plot = img.split('/')[-1].replace('_ortho.tif', '').replace('_plotclip.tif', '')
    plot_name = plot.replace('_', ' ')
    print(f'Image: {plot_name}')
    genotype = get_genotype(plot_name, args.geojson)
    a_img = open_image(img)
    df = pd.DataFrame()
    # myProj = Proj("+proj=utm +zone=12N, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    myProj = Proj("+proj=utm +zone=12 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    try:
        predictions = model.predict(a_img)
        labels, boxes, scores = predictions
        print(scores)
        copy = a_img.copy()

        for i, box in enumerate(boxes):
        
            cont_cnt += 1

            min_x, min_y, max_x, max_y = get_min_max(box)
            center_x, center_y = ((max_x+min_x)/2, (max_y+min_y)/2)
            if args.type in ['FLIR', 'RGB']: 
                lat, lon = pixel2geocoord(img, center_x, center_y)
                nw_lat, nw_lon = pixel2geocoord(img, min_x, max_y)
                se_lat, se_lon = pixel2geocoord(img, max_x, min_y)
                nw_e, nw_n = myProj(nw_lon, nw_lat) 
                se_e, se_n = myProj(se_lon, se_lat)

            elif args.type in ['DRONE']:
                northing, easting = pixel2geocoord(img, center_x, center_y)
                lon, lat = myProj(easting, northing, inverse=True)
                nw_n, nw_e = pixel2geocoord(img, min_x, max_y)
                se_n, se_e = pixel2geocoord(img, max_x, min_y)
                nw_lon, nw_lat = myProj(nw_e, nw_n, inverse=True)
                se_lon, se_lat = myProj(se_e, se_n, inverse=True)
                print(se_lon)

            area_sq = (se_e - nw_e) * (se_n - nw_n)
            lett_dict[cont_cnt] = {
                'date': args.date,
                'pred_conf': scores[i].detach().numpy(),
                'plot': plot,
                'genotype': genotype,
                'lon': lon,
                'lat': lat,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'nw_lat': nw_lat,
                'nw_lon': nw_lon,
                'se_lat': se_lat,
                'se_lon': se_lon,
                'bounding_area_m2': area_sq
            }

        df = pd.DataFrame.from_dict(lett_dict, orient='index', columns=['date',
                                                                    'pred_conf',
                                                                    'plot',
                                                                    'genotype',
                                                                    'lon',
                                                                    'lat',
                                                                    'min_x',
                                                                    'max_x',
                                                                    'min_y',
                                                                    'max_y',
                                                                    'nw_lat',
                                                                    'nw_lon',
                                                                    'se_lat',
                                                                    'se_lon',
                                                                    'bounding_area_m2']).set_index('date')
    except:
        pass

    return df


# --------------------------------------------------
def main():
    """Detect panicles here"""

    args = get_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    img_list = glob.glob(''.join([str(args.dir), os.path.sep, '*.tif']))
    print(img_list)

    major_df = pd.DataFrame()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:

        df = p.map(process_image, img_list)
        major_df = major_df.append(df)

    # for img in img_list:

    #     temp_df = process_image(img=img)
    #     major_df.append(temp_df)

    out_path = os.path.join(args.outdir, f'{args.date}_detection.csv')
    # major_df = pd.concat(major_df)
    major_df.to_csv(out_path)

    print(f'Done, see outputs in ./{args.outdir}.')
# --------------------------------------------------
if __name__ == '__main__':
    main()
