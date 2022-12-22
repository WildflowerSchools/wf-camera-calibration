import honeycomb_io
import cv_utils
import video_io
import cv2
import numpy as np
import datetime
import math
import os
import logging

logger = logging.getLogger(__name__)

def visualize_calibration(
    visualization_datetime,
    room_corners,
    floor_height=0.0,
    environment_id=None,
    environment_name=None,
    mark_device_locations=False,
    marked_device_types = ['UWBANCHOR', 'PI3WITHCAMERA', 'PI4WITHCAMERA', 'PIZEROWITHCAMERA'],
    num_grid_points_per_distance_unit=2,
    grid_point_radius=3,
    grid_point_color='#00ff00',
    grid_point_alpha=1.0,
    corner_label_horizontal_alignment='left',
    corner_label_vertical_alignment='bottom',
    corner_label_font_scale=1.0,
    corner_label_line_width=2,
    corner_label_color='#00ff00',
    corner_label_alpha=1.0,
    device_point_radius=3,
    device_point_color='#ff0000',
    device_point_alpha=1.0,
    device_label_horizontal_alignment='left',
    device_label_vertical_alignment='bottom',
    device_label_font_scale=1.0,
    device_label_line_width=2,
    device_label_color='#ff0000',
    device_label_alpha=1.0,
    local_image_directory='./images',
    image_filename_extension='png',
    local_video_directory='./videos',
    video_filename_extension=None,
    output_directory='./image_overlays',
    output_filename_extension='png',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if environment_id is None:
        if environment_name is None:
            raise ValueError('Must specify either environment ID or environment name')
        logger.info('Environment ID not specified. Fetching environmnt ID for environment name {}'.format(environment_name))
        environment_id = honeycomb_io.fetch_environment_id(
            environment_name=environment_name,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    logger.info('Visualizing calibration for environment id {}'.format(environment_id))
    logger.info('Generating object points for grid')
    floor_grid_object_points = generate_floor_grid_object_points(
        room_corners=room_corners,
        floor_height=0.0,
        num_points_per_distance_unit=num_grid_points_per_distance_unit
    )
    grid_corner_object_points = generate_grid_corner_object_points(
        room_corners,
        floor_height
    )
    if mark_device_locations:
        device_positions = honeycomb_io.fetch_device_positions(
            environment_id=environment_id,
            datetime=visualization_datetime,
            device_types=marked_device_types,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        device_names = list()
        device_object_points = list()
        for device_id, device_info in device_positions.items():
            device_names.append(device_info['name'])
            device_object_points.append(device_info['position'])
        device_object_points = np.asarray(device_object_points)
        logger.info('Fetched {} valid position assignments'.format(len(device_names)))
    logger.info('Fetching images')
    metadata = video_io.fetch_images(
        image_timestamps=[visualization_datetime],
        environment_id=environment_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        local_image_directory=local_image_directory,
        image_filename_extension=image_filename_extension,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension
    )
    logger.info('Fetched {} images'.format(len(metadata)))
    logger.info('Fetching camera calibrations')
    camera_ids = [metadatum['device_id'] for metadatum in metadata]
    camera_calibrations = honeycomb_io.fetch_camera_calibrations(
        camera_ids=camera_ids,
        start=visualization_datetime,
        end=visualization_datetime,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Fetching camera names')
    camera_names = honeycomb_io.fetch_camera_names(
        camera_ids=camera_ids,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    for metadatum in metadata:
        camera_device_id = metadatum.get('device_id')
        camera_name = camera_names[camera_device_id]
        logger.info('Visualizing calibration for camera {}'.format(camera_name))
        camera_calibration = camera_calibrations[camera_device_id]
        image_corners = np.array([
            [0.0, 0.0],
            [float(camera_calibration.get('image_width')), float(camera_calibration.get('image_height'))]
        ])
        logger.info('Calculating image points from object points')
        floor_grid_image_points = cv_utils.project_points(
            object_points=floor_grid_object_points,
            rotation_vector=camera_calibration.get('rotation_vector'),
            translation_vector=camera_calibration.get('translation_vector'),
            camera_matrix=camera_calibration.get('camera_matrix'),
            distortion_coefficients=camera_calibration.get('distortion_coefficients'),
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=image_corners
        )
        grid_corner_image_points = cv_utils.project_points(
            object_points=grid_corner_object_points,
            rotation_vector=camera_calibration.get('rotation_vector'),
            translation_vector=camera_calibration.get('translation_vector'),
            camera_matrix=camera_calibration.get('camera_matrix'),
            distortion_coefficients=camera_calibration.get('distortion_coefficients'),
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=image_corners
        )
        if mark_device_locations:
            device_image_points = cv_utils.project_points(
                object_points=device_object_points,
                rotation_vector=camera_calibration.get('rotation_vector'),
                translation_vector=camera_calibration.get('translation_vector'),
                camera_matrix=camera_calibration.get('camera_matrix'),
                distortion_coefficients=camera_calibration.get('distortion_coefficients'),
                remove_behind_camera=True,
                remove_outside_frame=True,
                image_corners=image_corners
            )
        logger.info('Drawing visualization')
        image = cv2.imread(metadatum.get('image_local_path'))
        image = draw_floor_grid_image_points(
            original_image=image,
            image_points=floor_grid_image_points,
            radius=grid_point_radius,
            color=grid_point_color,
            alpha=grid_point_alpha
        )
        image = draw_floor_grid_corner_labels(
            original_image=image,
            image_points=grid_corner_image_points,
            object_points=grid_corner_object_points,
            horizontal_alignment=corner_label_horizontal_alignment,
            vertical_alignment=corner_label_vertical_alignment,
            font_scale=corner_label_font_scale,
            line_width=corner_label_line_width,
            color=corner_label_color,
            alpha=corner_label_alpha
        )
        if mark_device_locations:
            image = draw_device_image_points(
                original_image=image,
                image_points=device_image_points,
                radius=device_point_radius,
                color=device_point_color,
                alpha=device_point_alpha
            )
            image = draw_device_labels(
                original_image=image,
                image_points=device_image_points,
                labels=device_names,
                horizontal_alignment=device_label_horizontal_alignment,
                vertical_alignment=device_label_vertical_alignment,
                font_scale=device_label_font_scale,
                line_width=device_label_line_width,
                color=device_label_color,
                alpha=device_label_alpha
            )
        logger.info('Saving visualization')
        output_filename = 'calibration_{}_{}.{}'.format(
            visualization_datetime.astimezone(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'),
            camera_name,
            output_filename_extension
        )
        output_path = os.path.join(output_directory, output_filename)
        os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(output_path, image)

def overlay_floor_lines(
    visualization_datetime,
    beginning_of_line,
    end_of_line,
    first_line_position,
    last_line_position,
    floor_height=0.0,
    line_direction='x',
    point_spacing=0.1,
    line_spacing=0.5,
    environment_id=None,
    line_point_radius=3,
    line_point_line_width = 1,
    line_point_color='#00ff00',
    line_point_alpha=1.0,
    local_image_directory='./images',
    image_filename_extension='png',
    local_video_directory='./videos',
    video_filename_extension='mp4',
    output_directory='./image_overlays',
    output_filename_extension='png',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Visualizing calibration for environment id {}'.format(environment_id))
    logger.info('Generating object points for lines')
    object_points = list()
    for line_position in np.linspace(
        start=first_line_position,
        stop=last_line_position,
        num=int(round((last_line_position - first_line_position)/line_spacing)) + 1,
        endpoint=True
    ):
        for point_position in np.linspace(
            start=beginning_of_line,
            stop=end_of_line,
            num=int(round((end_of_line - beginning_of_line)/point_spacing)) + 1,
            endpoint=True
        ):
            if line_direction == 'x':
                object_points.append([point_position, line_position, floor_height])
            elif line_direction == 'y':
                object_points.append([line_position, point_position, floor_height])
            else:
                raise ValueError('Line direction must be \'x\' or \'y\'')
    object_points = np.asarray(object_points)
    logger.info('Fetching images')
    metadata = video_io.fetch_images(
        image_timestamps=[visualization_datetime],
        environment_id=environment_id,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        local_image_directory=local_image_directory,
        image_filename_extension=image_filename_extension,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension
    )
    logger.info('Fetched {} images'.format(len(metadata)))
    logger.info('Fetching camera calibrations')
    camera_ids = [metadatum['device_id'] for metadatum in metadata]
    camera_calibrations = honeycomb_io.fetch_camera_calibrations(
        camera_ids=camera_ids,
        start=visualization_datetime,
        end=visualization_datetime,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Fetching camera names')
    camera_names = honeycomb_io.fetch_camera_names(
        camera_ids=camera_ids,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    for metadatum in metadata:
        camera_id = metadatum.get('device_id')
        camera_calibration = camera_calibrations[camera_id]
        camera_name = camera_names[camera_id]
        logger.info('Drawing lines for camera {}'.format(camera_name))
        logger.info('Calculating image points from object points')
        image_points = cv_utils.project_points(
            object_points=object_points,
            rotation_vector=camera_calibration.get('rotation_vector'),
            translation_vector=camera_calibration.get('translation_vector'),
            camera_matrix=camera_calibration.get('camera_matrix'),
            distortion_coefficients=camera_calibration.get('distortion_coefficients'),
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=[
                [0,0],
                [camera_calibration['image_width'], camera_calibration['image_height']]
            ]
        )
        logger.info('Drawing lines')
        image = cv2.imread(metadatum.get('image_local_path'))
        for point_index in range(image_points.shape[0]):
            point = image_points[point_index]
            if np.any(np.isnan(point)):
                continue
            image = cv_utils.draw_circle(
                original_image=image,
                coordinates=point,
                radius=line_point_radius,
                line_width=line_point_line_width,
                color=line_point_color,
                fill=True,
                alpha=line_point_alpha
            )
        logger.info('Saving visualization')
        output_filename = 'floor_lines_{}_{}.{}'.format(
            visualization_datetime.astimezone(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'),
            camera_name,
            output_filename_extension
        )
        output_path = os.path.join(output_directory, output_filename)
        os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(output_path, image)


def draw_floor_grid_image_points(
    original_image,
    image_points,
    radius=3,
    line_width=1.5,
    color='#00ff00',
    fill=True,
    alpha=1.0
):
    image_points = np.asarray(image_points).reshape((-1, 2))
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        point = image_points[point_index]
        if np.any(np.isnan(point)):
            continue
        output_image = cv_utils.draw_circle(
            original_image=output_image,
            coordinates=point,
            radius=radius,
            line_width=line_width,
            color=color,
            fill=True,
            alpha=1.0
        )
    return output_image

def draw_device_image_points(
    original_image,
    image_points,
    radius=3,
    line_width=1.5,
    color='#ff0000',
    fill=True,
    alpha=1.0
):
    image_points = np.asarray(image_points).reshape((-1, 2))
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        point = image_points[point_index]
        if np.any(np.isnan(point)):
            continue
        output_image = cv_utils.draw_circle(
            original_image=output_image,
            coordinates=point,
            radius=radius,
            line_width=line_width,
            color=color,
            fill=True,
            alpha=1.0
        )
    return output_image

def draw_floor_grid_corner_labels(
    original_image,
    image_points,
    object_points,
    horizontal_alignment='left',
    vertical_alignment='bottom',
    font_scale=1.0,
    line_width=2,
    color='#00ff00',
    alpha=1.0
):
    output_image = original_image.copy()
    for point_index in range(object_points.shape[0]):
        object_point = object_points[point_index]
        image_point = image_points[point_index]
        if np.any(np.isnan(image_point)):
            continue
        text = '({}, {})'.format(
            round(object_point[0]),
            round(object_point[1])
        )
        output_image = cv_utils.draw_text(
            original_image=output_image,
            anchor_coordinates=image_point,
            text=text,
            horizontal_alignment=horizontal_alignment,
            vertical_alignment=vertical_alignment,
            font_scale=font_scale,
            line_width=line_width,
            color=color,
            alpha=alpha
        )
    return output_image

def draw_device_labels(
    original_image,
    image_points,
    labels,
    horizontal_alignment='left',
    vertical_alignment='bottom',
    font_scale=1.0,
    line_width=1,
    color='#ff0000',
    alpha=1.0
):
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        image_point = image_points[point_index]
        if np.any(np.isnan(image_point)):
            continue
        text = labels[point_index]
        output_image = cv_utils.draw_text(
            original_image=output_image,
            anchor_coordinates=image_point,
            text=text,
            horizontal_alignment=horizontal_alignment,
            vertical_alignment=vertical_alignment,
            font_scale=font_scale,
            line_width=line_width,
            color=color,
            alpha=alpha
        )
    return output_image

def generate_grid_corner_object_points(
    room_corners,
    floor_height=0.0
):
    grid_corners = generate_grid_corners(room_corners)
    grid_corner_object_points=np.array([
        [grid_corners[0, 0], grid_corners[0, 1], floor_height],
        [grid_corners[0, 0], grid_corners[1, 1], floor_height],
        [grid_corners[1, 0], grid_corners[0, 1], floor_height],
        [grid_corners[1, 0], grid_corners[1, 1], floor_height]
    ])
    return grid_corner_object_points

def generate_floor_grid_object_points(
    room_corners,
    floor_height=0.0,
    num_points_per_distance_unit=2
):
    num_points_per_distance_unit = round(num_points_per_distance_unit)
    grid_corners = generate_grid_corners(room_corners)
    x_grid, y_grid = np.meshgrid(
        np.linspace(
            grid_corners[0, 0],
            grid_corners[1, 0],
            num=round(grid_corners[1, 0] - grid_corners[0, 0])*num_points_per_distance_unit + 1,
            endpoint=True
        ),
        np.linspace(
            grid_corners[0, 1],
            grid_corners[1, 1],
            num=round(grid_corners[1, 1] - grid_corners[0, 1])*num_points_per_distance_unit + 1,
            endpoint=True
            )
    )
    grid = np.stack((x_grid, y_grid, np.full_like(x_grid, floor_height)), axis=-1)
    object_points = grid.reshape((-1, 3))
    return object_points

def generate_grid_corners(room_corners):
    room_corners = np.asarray(room_corners)
    grid_corners = np.array([
        [float(math.ceil(room_corners[0, 0])), float(math.ceil(room_corners[0, 1]))],
        [float(math.floor(room_corners[1, 0])), float(math.floor(room_corners[1, 1]))],
    ])
    return grid_corners
