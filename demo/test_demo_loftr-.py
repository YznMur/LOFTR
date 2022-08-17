front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

os.sys.path.append("../")  # Add the project directory
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
try:
    from demo.utils import (AverageTimer, VideoStreamer,
                            make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LoFTR online demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight', type=str, help="Path to the checkpoint.")
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1280, 720],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--save_input', action='store_true',
        help='Save the input images to a video (for gathering repeatable input source).')
    parser.add_argument(
        '--skip_frames', type=int, default=1, 
        help="Skip frames from webcam input.")
    parser.add_argument(
        '--top_k', type=int, default=20000, help="The max vis_range (please refer to the code).")
    parser.add_argument(
        '--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")

    opt = parser.parse_args()
    # print(front_matter)
    parser.print_help()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError("GPU is required to run this demo.")

    # Initialize LoFTR
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
    matcher = matcher.eval().to(device=device)

    # Configure I/O
    if opt.save_video:
        print('Writing video to loftr-matches.mp4...')
        writer = cv2.VideoWriter('loftr-matches.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640*2 + 10, 480))
    if opt.save_input:
        print('Writing video to demo-input.mp4...')
        input_writer = cv2.VideoWriter('demo-input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (960, 540))

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    # frame, ret, resize_scale = vs.next_frame()
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        window_name = 'LoFTR Matches'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the reference image (left)\n'
          '\td/f: move the range of the matches (ranked by confidence) to visualize\n'
          '\tc/v: increase/decrease the length of the visualization range (i.e., total number of matches) to show\n'
          '\tq: quit')
    timer = AverageTimer()
    vis_range = [opt.bottom_k, opt.top_k]
    index=0
    while True:
        frame_id = 0  
        last_image_id = 0
        frame_tensor = frame2tensor(frame, device)
        # print(frame, type(frame))
        im_src = frame
        last_data = {'image0': frame_tensor}
        last_frame = frame
        frame_id += 1
        print("frame=",frame_id)
        # frame, ret, resize_scale = vs.next_frame()
        frame, ret = vs.next_frame()
        if frame_id % opt.skip_frames != 0:
            # print("Skipping frame.")
            continue
        if opt.save_input:
            inp = np.stack([frame]*3, -1)
            inp_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            input_writer.write(inp_rgb)
        if not ret:
            print('Finished demo_loftr.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        last_data = {**last_data, 'image1': frame_tensor}
        matcher(last_data)

        total_n_matches = len(last_data['mkpts0_f'])
        mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]
        # np.savetxt('poses/pos_img_' + str(index)+'__match_imgs'+ str(index+1) + '&' + str(index), mkpts0 * resize_scale)
        # np.savetxt('poses/pos_img_' + str(index + 1)+'__match_imgs'+ str(index+1) + '&' + str(index), mkpts1 * resize_scale)
        # np.savetxt('confidence/conf_match_imgs' + str(index+1) + '&' + str(index), mconf)
            
        
        sift = cv2.xfeatures2d.SIFT_create() 
        kp_image, desc_image =sift.detectAndCompute(im_src, None) 
        index_params = dict(algorithm = 0, trees = 5) 
        search_params = dict() 
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp_grayframe, desc_grayframe = sift.detectAndCompute(last_frame, None) 
        matches= flann.knnMatch(desc_image, desc_grayframe, k=2) 
        good_points=[] 
        # print(kp_grayframe)
        for m, n in matches: 
            
            if(m.distance < 0.6*n.distance): 
                good_points.append(m)

        # query_pts = np.float32([kp_image[m.queryIdx] 
        #         .pt for m in good_points]).reshape(-1, 1, 2) 
        
        # train_pts = np.float32([kp_grayframe[m.trainIdx] 
        #                 .pt for m in good_points]).reshape(-1, 1, 2) 

        query_pts = np.float32(mkpts0).reshape(-1, 1, 2) 
        
        train_pts = np.float32(mkpts1).reshape(-1, 1, 2) 
        # print(query_pts)
       
        matrix, mask = cv2.findHomography(query_pts, train_pts) 

        matches_mask = mask.ravel().tolist() 
        
        h,w = im_src.shape
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        print(pts)
        
        dst = cv2.perspectiveTransform(pts, matrix)
        
        homography = cv2.polylines(last_frame, [np.int32(dst)], True, (255, 0, 0), 3) 
 
        
        # pts_src = np.array(mkpts0, dtype=np.float32)
        # pts_dst = np.array(mkpts1, dtype=np.float32)

        # pts_src = np.array([[281, 238],[325, 297], [283, 330],[248, 325],[213, 321]], dtype=np.float32)
        # pts_dst = np.array([[377, 251],[377, 322],[316, 315],[289, 284],[263,255]], dtype=np.float32)

        # pts_src = np.array([[325, 297], [283, 330],[248, 325],[213, 321]], dtype=np.float32)
        # pts_dst = np.array([[377, 322],[316, 315],[289, 284],[263,255]], dtype=np.float32)


        # pts = np.float32([ [0,0],[0,h/2-1],[w/2-1,h/2-1],[w/2-1,0]]).reshape(-1,1,2)
        # pts1 = np.float32([ [0,0],[0,h_l-1],[w_l-1,h_l-1],[w_l-1,0]]).reshape(-1,1,2)
        # perspective_transform = cv2.getPerspectiveTransform(pts, pts1)    

        # dst = cv2.perspectiveTransform(pts, perspective_transform)
        
        # homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 
        
        # cv2.imshow("Homography", homography) 

        # im_dst = cv2.warpPerspective(im_src, h, size)
        # img_warp = cv2.warpPerspective(im_src, h, (last_frame.shape[1],last_frame.shape[0]))
        # print(img_warp)
        # out_img = np.array([])
        # good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(total_n_matches)]

        # display = cv2.drawMatches(im_src, mkpts0, last_frame, mkpts1, good_matches,
        #                           None,
        #                           matchColor=match_color,
        #                           singlePointColor=pt_color,
        #                           matchesMask=mask.ravel().tolist(), flags=4)
        # img_warp = cv2.drawMatches(im_src, mkpts0, last_frame, mkpts1, matches1to2=good_matches, outImg=out_img)
        # out_img = cv2.drawMatches(im_src, cv_kp1, last_frame, cv_kp2, matches1to2=good_matches, outImg=out_img)

        # index += 1

        # Normalize confidence.
        if len(mconf) > 0:
            conf_vis_min = 0.
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

        timer.update('forward')
        alpha = 0
        color = cm.jet(mconf, alpha=alpha)

        text = [
            f'LoFTR',
            '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
        ]
        small_text = [
            f'Showing matches from {vis_range[0]}:{vis_range[1]}',
            f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints= True, small_text=small_text)

        # Save high quality png, optionally with dynamic alpha support (unreleased yet).
        # save_path = 'demo_vid/{:06}'.format(frame_id)
        # make_matching_plot(
        #     last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
        #     path=save_path, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            if opt.save_video:
                writer.write(out)
            cv2.imshow('LoFTR Matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                if opt.save_video:
                    writer.release()
                if opt.save_input:
                    input_writer.release()
                vs.cleanup()
                print('Exiting...')
                break
            elif key == 'n':  
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
                frame_id_left = frame_id
            elif key in ['d', 'f']:
                if key == 'd':
                    if vis_range[0] >= 0:
                       vis_range[0] -= 200
                       vis_range[1] -= 200
                if key =='f':
                    vis_range[0] += 200
                    vis_range[1] += 200
                print(f'\nChanged the vis_range to {vis_range[0]}:{vis_range[1]}')
            elif key in ['c', 'v']:
                if key == 'c':
                    vis_range[1] -= 50
                if key =='v':
                    vis_range[1] += 50
                print(f'\nChanged the vis_range[1] to {vis_range[1]}')
        elif opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            warp_out = 'homography_{:06}_{:06}'.format(stem0, stem1)
            # out_img_with_matching = 'out_img{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            img_warp_out = str(Path(opt.output_dir, warp_out + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
            cv2.imwrite(img_warp_out, homography)
            # cv2.imwrite(out_img_with_matching, homography)
        else:
            raise ValueError("output_dir is required when no display is given.")
        timer.update('viz')
        timer.print()


    cv2.destroyAllWindows()
    vs.cleanup()
