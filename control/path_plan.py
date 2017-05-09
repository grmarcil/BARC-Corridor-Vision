import sys
import cv2
import numpy as np
import skimage
import skimage.io
import skimage.color
import numpy as np
import scipy.ndimage
import scipy.interpolate
import imutils
import matplotlib.pyplot as plt
import matplotlib.animation


# Path Planning
###############
def robot_to_map_perspective(img):
    w = 480
    h = 640
    M = np.load("perspTransformMatrix.npy")
    map_img = cv2.warpPerspective(img, M, (w, h))
    return map_img


def get_biggest_contour(contours):
    areas = [cv2.contourArea(ctr) for ctr in contours]
    idx = np.argmax(areas)
    ctr = contours[idx]
    # Reshape contour to get rid of extra nesting dimension
    # ie just return a list of [x, y] points
    pts = len(ctr)
    ctr = ctr.reshape((pts, 2))
    return ctr


def compute_path(map_img):
    # This may need considerable tuning/reworking with real noisy data
    im2, contours, hierarchy = cv2.findContours(map_img, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    floor_ctr = get_biggest_contour(contours)

    # Robot is at the bottom center of the image (240,640), so find max y coord
    # on contour to begin path into free floor space
    path = [[240, 639]]
    nearest_y = max(floor_ctr[:, 1])
    # extract min and max x along every pixel scanline from nearest_y to top of
    # map image
    for y in range(nearest_y, 0, -1):
        # Get contour indices of all floor points with this y value
        floor_on_scanline = np.argwhere(floor_ctr[:, 1] == y)
        floor_on_scanline = floor_on_scanline.reshape(len(floor_on_scanline))
        # Extract these points
        scanline_pts = floor_ctr[floor_on_scanline.tolist()]
        min_x = np.min(scanline_pts[:, 0])
        max_x = np.max(scanline_pts[:, 0])
        mid_x = (max_x - min_x) / 2.0 + min_x
        # Add midpoint of free x space to path
        path.append([mid_x, y])
    return np.asarray(path)


def smooth_path(path):
    # Apply gaussian smoothing to path x coordinates
    sigma = 25
    path[:, 0] = scipy.ndimage.filters.gaussian_filter1d(path[:, 0], sigma)
    return path


def render_map_with_path(map_img, path):
    map_img = skimage.color.gray2rgb(map_img)
    path = np.round(path).astype(int)
    path_color = (255, 0, 0)
    for i in range(len(path) - 1):
        pt = path[i]
        pt2 = path[i + 1]
        cv2.line(map_img, tuple(pt), tuple(pt2), path_color, 2)
    # skimage.io.imshow(map_img)
    # skimage.io.show()
    return map_img


# Point manipulation tools
##########################
def img_to_robot_coords(img_pt):
    # robot world origin (0,0) fixed at bottom center of image (240, 639)
    robot_pt = [img_pt[0] - 240, 639 - img_pt[1]]
    return robot_pt


def robot_to_img_coords(robot_pt):
    img_pt = [240 + robot_pt[0], 639 - robot_pt[1]]
    return img_pt


def convert_path_to_robot_frame(path):
    new_path = [img_to_robot_coords(pt) for pt in path]
    return np.array(new_path)


def convert_path_to_img_frame(path):
    new_path = [robot_to_img_coords(pt) for pt in path]
    return np.array(new_path)


def interpolate_path(path):
    # Construct path with evenly spaced x,y points along its length

    # Build path distance array
    dist_arr = np.zeros(len(path))
    for i in range(len(path) - 1):
        step = np.linalg.norm(path[i + 1] - path[i])
        dist_arr[i + 1] = dist_arr[i] + step

    # Interpolation functions for x and y along path distance
    x_arr = path[:, 0]
    y_arr = path[:, 1]
    interp_x = scipy.interpolate.interp1d(dist_arr, x_arr)
    interp_y = scipy.interpolate.interp1d(dist_arr, y_arr)

    # New step distance will be one pixel unit
    new_dist_arr = np.arange(0, dist_arr[len(path) - 1])
    new_x_arr = interp_x(new_dist_arr)
    new_y_arr = interp_y(new_dist_arr)

    new_path = np.zeros((len(new_dist_arr), 2))
    new_path[:, 0] = new_x_arr
    new_path[:, 1] = new_y_arr
    return new_path


def path_with_psi(path):
    new_path = np.zeros((len(path), 3))
    new_path[:, 0:2] = path
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        psi = np.arctan2(dy, dx)
        new_path[i, 2] = psi
    new_path[-1, 2] = new_path[-2, 2]
    return new_path


# Controls
##########
def follow_path(path):
    # Initial state, bottom center of image, facing straight ahead
    # (x, y, psi)
    state0 = np.array([0, 0, np.pi / 2])
    state = state0
    state_hist = [state0]
    # Constant velocity 48 px/s (12 in/s at 4ppi)
    v = 48
    # Timestep and simulation length
    dt = 0.05
    Lsim = 260

    # KDTree for finding nearest path point
    path_kdtree = scipy.spatial.KDTree(path)

    for k in range(Lsim):
        # Lookup path point closest to robot state
        path_pt_idx = path_kdtree.query(state)[1]
        path_pt = path[path_pt_idx]

        beta = p_controller(state, path_pt)
        state = dynamics_step(state, beta, v, dt)
        state_hist.append(state)

    return np.array(state_hist)


def p_controller(state, state_ref):
    # Compute steering angle proportional to deviation in position and angle
    # from reference path
    # Controller constants
    # 0.03 and -0.07 work pretty well
    k_pos = 0.03
    k_psi = -0.07
    # Compute errors
    e = state - state_ref
    # Calculate perpendicular dist to path, preserving sign of e_x
    e_pos = np.sign(e[0]) * np.sqrt(e[0] * e[0] + e[1] * e[1])
    # May need to do some wrapping
    e_psi = e[2]
    # Compute control action
    beta = k_pos * e_pos + k_psi * e_psi
    return beta


def dynamics_step(state, beta, v, dt):
    # 1/2 wheelbase
    lr = 5.25
    # Calculate derivatives
    dx = v * np.cos(state[2] + beta)
    dy = v * np.sin(state[2] + beta)
    dpsi = v / lr * np.sin(beta)

    new_state = [state[0] + dt * dx, state[1] + dt * dy, state[2] + dt * dpsi]
    return new_state


def plot_trajectory(state_trajectory, ref_path):
    plt.figure()
    plt.plot(state_trajectory[:, 0], state_trajectory[:, 1])
    plt.plot(ref_path[:, 0], ref_path[:, 1])
    plt.show()


def animate_trajectory(state_traj, map_path_img):
    frames = []
    # load overhead car sprite with alpha channel
    car = cv2.imread("car_overhead.png", -1)
    car_h = car.shape[0]
    car_w = car.shape[1]

    for k in range(len(state_traj)):
        states = state_traj[0:k+1]
        frame = map_path_img.copy()
        frame = render_frame(states, frame, car)
        frames.append(frame)

    fig = plt.figure()
    im = plt.imshow(frames[0])

    # func to advance frame
    def update_frame(j):
        im.set_array(frames[j])
        return [im]

    anim = matplotlib.animation.FuncAnimation(fig, update_frame, frames=range(len(frames)), interval=50, blit=True)
    anim.save("2barc0380.mp4")
    # plt.show()


def render_frame(states, frame, car):
    state = states[-1]
    pad = car.shape[1] + 10
    img_pt = robot_to_img_coords(state[0:2])
    x = int(img_pt[0])
    y = int(img_pt[1])
    psi = state[2]*180/np.pi

    # trace trajectory so far
    states = np.round(states).astype(int)
    path_color = (0, 0, 255)
    for i in range(len(states) - 1):
        pt = robot_to_img_coords(states[i][0:2])
        pt2 = robot_to_img_coords(states[i + 1][0:2])
        cv2.line(frame, tuple(pt), tuple(pt2), path_color, 2)

    # rotate car sprite
    rotate_ang = -(psi - 90)
    rot_car = imutils.rotate_bound(car, rotate_ang)
    h = rot_car.shape[0]
    w = rot_car.shape[1]
    half_h = int(rot_car.shape[0] / 2)
    half_w = int(rot_car.shape[1] / 2)
    # sprite corner coordinates
    corner_x = x - half_w
    corner_y = y - half_h

    # add padding in case sprite goes outside image
    frame = np.pad(frame, [(pad, pad), (pad, pad), (0, 0)], 'edge')
    # account for padding in sprite coordinates
    x_min = pad + corner_x
    x_max = pad + w + corner_x
    y_min = pad + corner_y
    y_max = pad + h + corner_y

    for c in range(0, 3):
        frame[y_min:y_max, x_min:x_max, c] = rot_car[:, :, c] * (
            rot_car[:, :, 3] / 255.0) + frame[y_min:y_max, x_min:x_max, c] * (
                1.0 - rot_car[:, :, 3] / 255.0)

    # Trim to remove padding
    frame = frame[pad:-pad][pad:-pad][:]
    return frame


def main(img_file):
    img = skimage.io.imread(img_file)
    map_img = robot_to_map_perspective(img)

    path = compute_path(map_img)
    path = convert_path_to_robot_frame(path)
    path = interpolate_path(path)
    path = smooth_path(path)
    map_path_img = render_map_with_path(map_img,
                                        convert_path_to_img_frame(path))
    ref_path = path_with_psi(path)
    state_trajectory = follow_path(ref_path)
    # plot_trajectory(state_trajectory, ref_path)
    animate_trajectory(state_trajectory, map_path_img)


if __name__ == '__main__':
    main(sys.argv[1])
