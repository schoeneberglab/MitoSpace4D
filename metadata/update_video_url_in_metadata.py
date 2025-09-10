import argparse
import json
import pubchempy as pcp
import os.path as osp


if __name__ == '__main__':
    proj_dir = "/u/earkfeld/MitoSpace4D/"
    save_dir = osp.join(proj_dir, "runs", "lightning_logs", 'resnetbilstm_encoded_normal')
    metadata = json.load(open(f"{save_dir}/metadata.json"))

    # parse the whole metadata and for the 'images' field, replace the video URLs with the new ones
    for i in range(len(metadata['points'])):
        combined_vid_url = metadata['points'][i]['videos'][0]

        # replace the old URLs with the new ones
        mito_vid_url = combined_vid_url.replace("combined_", "mtg_")
        tmrm_vid_url = combined_vid_url.replace("combined_", "tmrm_")


        # metadata['points'][i]['images'][0] = combined_vid_url

        # delete the second URL
        # del metadata['points'][i]['images'][1]

        # delete the 'images' field and instead add a new field 'video_url'
        metadata['points'][i]['videos'] = [mito_vid_url, tmrm_vid_url]
        # del metadata['points'][i]['images']


    # save the metadata
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)