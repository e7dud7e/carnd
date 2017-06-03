def draw_aerial(img, lane, visualize=False):
    """
    img: the warped aerial view
    lane: object containing fitted line results
    """
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255 #the shape is height (720) x width (1280) x 3 copies of img
    
    #color left detected line points as red, right detected line points as blue
    out_img[lane.get_left_line().get_all_y(), lane.get_left_line().get_all_x()] = [255, 0, 0]
    out_img[lane.get_right_line.get_all_y(), lane.get_right_line.get_all_x()] = [0, 0, 255]
    
    #color fitted line as green
    out_img[lane.get_left_line().get_ploty(), lane.get_left_line().get_bestx()] = [0, 255, 0]
    out_img[lane.get_right_line().get_ploty(), lane.get_right_line().get_bestx()] = [0, 255, 0]
    if visualize:
        plt.imshow(out_img)        
    return out_img