import camera_calibration.analyze
import honeycomb_io
import cv_utils
import video_io
import pandas as pd
import numpy as np
import shutil
import re
import os
import logging

logger = logging.getLogger(__name__)

CALIBRATION_DATA_RE = r'(?P<colmap_image_id>[0-9]+) (?P<qw>[-0-9.]+) (?P<qx>[-0-9.]+) (?P<qy>[-0-9.]+) (?P<qz>[-0-9.]+) (?P<tx>[-0-9.]+) (?P<ty>[-0-9.]+) (?P<tz>[-0-9.]+) (?P<colmap_camera_id>[0-9]+) (?P<image_path>.+)'

RECOGNIZED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

def prepare_colmap_inputs(
    calibration_directory=None,
    calibration_identifier=None,
    image_info_path=None,
    images_directory_path=None,
    ref_images_data_path=None,
    additional_images=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_image_directory='./images',
    image_filename_extension='png',
    local_video_directory='./videos',
    video_filename_extension='mp4'
):
    """
    Prepares COLMAP input files based on data in local files and in Honeycomb.

    The script pulls calibration images from Honeycomb for each camera. To do
    so, it requires a CSV file which lists each camera along with its camera
    type (can be any string; used to group images by their intrinsic
    calibration parameters) and the timestamp from which the image should be
    drawn for that camera (can be different for different cameras). Column names
    should be device_id, camera_type, and image_timestamp (all other columns are
    ignored).

    Each image_timestamp should be a string in ISO format (or any format
    consumable by pandas.to_datetime()).

    The script will also attempt to pull position information from Honeycomb for
    each camera and include this information in a reference images file for
    COLMAP to reference during model alignment.

    User can also specify supply additional calibration images directly (e.g.,
    "fill-in" images taken with a portable camera to assist feature matching
    across cameras). These are specified in a dictionary. For each entry, the
    key is the camera type (again, an arbitrary string for grouping images by
    their intrinsic calibration parameters) and the value is the path to the
    directory containing the images. All recognized image files will be pulled
    from the specified directory. Currently, these are any files with a png,
    jpg, jpeg, or gif extension.

    By default, the script assumes all input and output files are in a single
    directory with path calibration_directory/calibration_identifier. It looks
    for a file image_info.csv in this directory. It creates a subdirectory
    called images under this directory with all of the calibration images and a
    file ref_images.txt with the camera position information (for use during
    model alignment). For images pulled from Honeycomb, the image filename stem
    is the camera device ID. These are the also the default path and naming
    conventions for COLMAP and for fetch_colmap_output_data_local().
    Alternatively, the user can explicitly specify the paths for each of these
    files.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        coordinate_space_id (str): The Honeycomb coordinate space ID for extrinsic calibration and position assignment objects
        image_info_path (str): Explicit path for CSV file containing image info (default is None)
        images_directory_path (str): Explicit path for directory to contain calibration images (default is None)
        ref_images_data_path (str): Explicit path for file to contain camera position info for model alignment (default is None)
        additional_images (dict): Information on additional calibration images to include (see above) (default is None)
        chunk_size (int): Number of objects to request at a time when querying Honeycomb (default is 100)
        client (MinimalHoneycombClient): Client object to use when connecting with Honeycomb (default is None)
        uri (str): URI to use for Honeycomb client (default is None)
        token_uri (str): Token URI to use for Honeycomb client (default is None)
        audience (str): Audience to use for Honeycomb client (default is None)
        client_id (str): Client ID to use for Honeycomb client (default is None)
        client_secret (str): Client secret to use for Honeycomb client (default is None)
        local_image_directory (str): Path where local copies of Honeycomb images should be stored (default is ./images)
        image_filename_extension (str): Filename extension for local copies of Honeycomb images (default is png)
        local_video_directory (str): Path where local copies of Honeycomb videos should be stored (default is ./videos)
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is mp4)
    """
    # Set input and output paths
    if image_info_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image info path or calibration directory and calibration identifier')
        image_info_path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'image_info.csv'
        )
    if images_directory_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image directory path or calibration directory and calibration identifier')
        images_directory_path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'images'
        )
    if ref_images_data_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either ref image data path or calibration directory and calibration identifier')
        ref_images_data_directory = os.path.join(
            calibration_directory,
            calibration_identifier
        )
        ref_images_data_filename = 'ref_images.txt'
    else:
        ref_images_data_directory = os.path.dirname(os.path.normpath(ref_images_data_path))
        ref_images_data_filename = os.path.basename(os.path.normpath(ref_images_data_path))
    # Fetch image info from CSV file
    image_info_columns = [
        'device_id',
        'camera_type',
        'image_timestamp'
    ]
    image_info_df = pd.read_csv(image_info_path)
    if not set(image_info_columns).issubset(set(image_info_df.columns)):
        raise ValueError('Image info CSV data must contain the following columns: {}'.format(
            image_info_columns
        ))
    image_info_df['image_timestamp'] = pd.to_datetime(image_info_df['image_timestamp'])
    ref_images_lines = list()
    for index, camera in image_info_df.iterrows():
        camera_device_id = camera['device_id']
        image_timestamp = camera['image_timestamp']
        camera_type = camera['camera_type']
        image_metadata = video_io.fetch_images(
            image_timestamps=[image_timestamp],
            camera_device_ids=[camera_device_id],
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
        if len(image_metadata) > 1:
            raise ValueError('More than one image returned for this camera and timestamp')
        image_info = image_metadata[0]
        source_path = image_info['image_local_path']
        # Copy image file
        output_directory = os.path.join(
            images_directory_path,
            camera_type
        )
        output_filename = '{}.{}'.format(
            camera_device_id,
            image_filename_extension
        )
        output_path = os.path.join(
            output_directory,
            output_filename
        )
        os.makedirs(output_directory, exist_ok=True)
        shutil.copy2(source_path, output_path)
        # Fetch camera position
        position = honeycomb_io.fetch_device_position(
            device_id=camera_device_id,
            datetime=image_timestamp,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        if position is not None:
            ref_images_line = ' '.join([
                os.path.join(
                    camera_type,
                    output_filename
                ),
                str(position[0]),
                str(position[1]),
                str(position[2])
            ])
            ref_images_lines.append(ref_images_line)
    os.makedirs(ref_images_data_directory, exist_ok=True)
    ref_images_path = os.path.join(
        ref_images_data_directory,
        ref_images_data_filename
    )
    with open(ref_images_path, 'w') as fp:
        fp.write('\n'.join(ref_images_lines))
    if additional_images is not None:
        for camera_type, input_directory in additional_images.items():
            # Copy image files
            output_directory = os.path.join(
                images_directory_path,
                camera_type
            )
            os.makedirs(output_directory, exist_ok=True)
            for dir_entry in os.scandir(input_directory):
                if not dir_entry.is_file():
                    continue
                if not os.path.splitext(dir_entry.name)[1][1:] in RECOGNIZED_IMAGE_EXTENSIONS:
                    continue
                input_filename = dir_entry.name
                output_filename = input_filename
                source_path = os.path.join(
                    input_directory,
                    input_filename
                )
                output_path = os.path.join(
                    output_directory,
                    output_filename
                )
                shutil.copy2(source_path, output_path)

def generate_image_info_file(
    calibration_directory=None,
    calibration_identifier=None,
    image_info_path=None,
    environment_id=None,
    environment_name=None,
    default_image_timestamp=None,
    default_camera_type=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    # Set output directory and path
    if image_info_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image info path or calibration directory and calibration identifier')
        output_directory = os.path.join(
            calibration_directory,
            calibration_identifier
        )
        image_info_path = os.path.join(
            output_directory,
            'image_info.csv'
        )
    else:
        output_directory = os.path.dirname(os.path.normpath(image_info_path))
    if default_image_timestamp is None:
        default_image_timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    camera_info_df = honeycomb_io.fetch_camera_info(
        environment_id=environment_id,
        environment_name=environment_name,
        start=default_image_timestamp,
        end=default_image_timestamp,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    image_info_df = (
        camera_info_df
        .reindex(columns=[
            'device_name',
            'part_number'
        ])
        .rename(columns={
            'device_name': 'camera_name',
            'part_number': 'camera_type'
        })
    )
    if default_camera_type is not None:
        image_info_df['camera_type'] = default_camera_type
    image_info_df['image_timestamp'] = default_image_timestamp.isoformat()
    os.makedirs(output_directory, exist_ok=True)
    image_info_df.to_csv(image_info_path)

def fetch_colmap_output_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    image_data_path=None,
    camera_data_path=None,
    ref_images_data_path=None
):
    """
    Fetches data from COLMAP input and output files and assembles into dataframe.

    The script essentially executes fetch_colmap_image_data_local(),
    fetch_colmap_camera_data_local(), and
    fetch_colmap_reference_image_data_local(); joins their outputs; calculates
    the difference between the camera position inputs and the camera position
    outputs; and assembles everything into a dataframe.

    For details, see documentation for the constituent functions.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        image_data_path (str): Explicit path for COLMAP images output file (default is None)
        camera_data_path (str): Explicit path for COLMAP cameras output file (default is None)
        ref_images_data_path (str): Explicit path for COLMAP ref images input file (default is None)

    Returns:
        (DataFrame) Dataframe containing COLMAP output data
    """
    # Fetch COLMAP image output
    df = fetch_colmap_image_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=image_data_path
    )
    # Fetch COLMAP cameras output
    cameras_df = fetch_colmap_camera_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=camera_data_path
    )
    df = df.join(cameras_df, on='colmap_camera_id')
    # Fetch COLMAP ref images input
    ref_images_df = fetch_colmap_reference_image_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=ref_images_data_path
    )
    df = df.join(ref_images_df, on='image_path')
    # Calculate fields
    df['image_path'] = df['image_path'].astype('string')
    df['position_error'] = df['position'] - df['position_input']
    df['position_error_distance'] = df['position_error'].apply(np.linalg.norm)
    return df

def fetch_colmap_image_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    """
    Fetches data from COLMAP images output file and assembles into dataframe.

    The script parses the COLMAP images output file, extracting the COLMAP
    image ID, COLMAP camera ID, image path, quaternion vector, and translation
    vector for each image.

    For each image, it then calculates a rotation vector from the quaternion
    vector; calculates a camera position from the rotation vector and
    translation vector; parses the image path into its subdirectory, filename
    stem, and filename extension; and fetches camera names from Honeycomb (if
    image filename stem matches a recognized camera device ID).

    By default, the script assumes that the COLMAP images output is in a file
    called images.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP and for prepare_colmap_inputs().
    Alternatively, the user can explicitly specify the path for the COLMAP
    images output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP image output file (default is None)
        chunk_size (int): Number of objects to request at a time when querying Honeycomb (default is 100)
        client (MinimalHoneycombClient): Client object to use when connecting with Honeycomb (default is None)
        uri (str): URI to use for Honeycomb client (default is None)
        token_uri (str): Token URI to use for Honeycomb client (default is None)
        audience (str): Audience to use for Honeycomb client (default is None)
        client_id (str): Client ID to use for Honeycomb client (default is None)
        client_secret (str): Client secret to use for Honeycomb client (default is None)

    Returns:
        (DataFrame) Dataframe containing image data
    """
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'images.txt'
        )
    data_list = list()
    with open(path, 'r') as fp:
        for line in fp.readlines():
            m = re.match(CALIBRATION_DATA_RE, line)
            if m:
                data_list.append({
                    'colmap_image_id': int(m.group('colmap_image_id')),
                    'quaternion_vector': np.asarray([
                        float(m.group('qw')),
                        float(m.group('qx')),
                        float(m.group('qy')),
                        float(m.group('qz'))
                    ]),
                    'translation_vector': np.asarray([
                        float(m.group('tx')),
                        float(m.group('ty')),
                        float(m.group('tz'))
                    ]),
                    'colmap_camera_id': int(m.group('colmap_camera_id')),
                    'image_path': m.group('image_path')

                })
    df = pd.DataFrame(data_list)
    df['rotation_vector'] = df['quaternion_vector'].apply(cv_utils.quaternion_vector_to_rotation_vector)
    df['position'] = df.apply(
        lambda row: cv_utils.extract_camera_position(
            row['rotation_vector'],
            row['translation_vector']
        ),
        axis=1
    )
    df['image_directory'] = df['image_path'].apply(lambda x: os.path.dirname(os.path.normpath(x))).astype('string')
    df['image_name'] = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(os.path.normpath(x)))[0]).astype('string')
    df['image_extension'] = df['image_path'].apply(
        lambda x: os.path.splitext(os.path.basename(os.path.normpath(x)))[1][1:]
        if len(os.path.splitext(os.path.basename(os.path.normpath(x)))[1]) > 1
        else None
    ).astype('string')
    logger.info('Attempting to extract camera device IDs from image names')
    df['device_id'] = df['image_name'].apply(honeycomb_io.extract_honeycomb_id).astype('object')
    device_ids = df['device_id'].dropna().unique().tolist()
    logger.info('Found {} device IDs among image names'.format(
        len(device_ids)
    ))
    logger.info('Fetching camera names')
    camera_names = honeycomb_io.fetch_camera_names(
        camera_ids=device_ids,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    df = df.join(
        pd.Series(camera_names, name='camera_name'),
        on='device_id'
    )
    df.set_index('colmap_image_id', inplace=True)
    df = df.reindex(columns=[
        'image_path',
        'image_directory',
        'image_name',
        'device_id',
        'camera_name',
        'image_extension',
        'colmap_camera_id',
        'quaternion_vector',
        'rotation_vector',
        'translation_vector',
        'position'
    ])
    return df

def fetch_colmap_camera_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None
):
    """
    Fetches data from COLMAP cameras output file and assembles into dataframe.

    The script parses the COLMAP cameras output file, extracting the COLMAP
    camera ID, COLMAP camera model (e.g., OPENCV), image width, image height,
    and intrinsic calibration parameters for each camera.

    For each camera, it then extracts the camera matrix and distortion
    coefficients from the intrinsic calibration parameters.

    By default, the script assumes that the COLMAP cameras output is in a file
    called cameras.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP and for prepare_colmap_inputs().
    Alternatively, the user can explicitly specify the path for the COLMAP
    cameras output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP cameras output file (default is None)

    Returns:
        (DataFrame) Dataframe containing camera data
    """
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either camera data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'cameras.txt'
        )
    cameras=list()
    with open(path, 'r') as fp:
        for line_index, line in enumerate(fp):
            if len(line) == 0 or line[0] == '#':
                continue
            word_list = line.split()
            if len(word_list) < 5:
                raise ValueError('Line {} is shorter than expected: {}'.format(
                    line_index,
                    line
                ))
            camera = {
                'colmap_camera_id': int(word_list[0]),
                'colmap_camera_model': word_list[1],
                'image_width': int(word_list[2]),
                'image_height': int(word_list[3]),
                'colmap_parameters': np.asarray([float(parameter_string) for parameter_string in word_list[4:]])
            }
            cameras.append(camera)
    df = pd.DataFrame.from_records(cameras)
    df['camera_matrix'] = df.apply(
        lambda row: colmap_parameters_to_opencv_parameters(
            row['colmap_parameters'],
            row['colmap_camera_model']
        )[0],
        axis=1
    )
    df['distortion_coefficients'] = df.apply(
        lambda row: colmap_parameters_to_opencv_parameters(
            row['colmap_parameters'],
            row['colmap_camera_model']
        )[1],
        axis=1
    )
    df = df.astype({
        'colmap_camera_id': 'int',
        'colmap_camera_model': 'string',
        'image_width': 'int',
        'image_height': 'int',
        'colmap_parameters': 'object',
        'camera_matrix': 'object',
        'distortion_coefficients': 'object'
    })
    df.set_index('colmap_camera_id', inplace=True)
    df = df.reindex(columns=[
        'colmap_camera_model',
        'image_width',
        'image_height',
        'colmap_parameters',
        'camera_matrix',
        'distortion_coefficients'
    ])
    return df

def fetch_colmap_reference_image_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None
):
    """
    Fetches data from COLMAP ref images input file and assembles into dataframe.

    The script parses the COLMAP ref images input file, extracting the image
    path and (input) camera position for each image (for comparison with the
    calculated camera position).

    By default, the script assumes that the COLMAP ref images input data is in a
    file called ref_images.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP and for prepare_colmap_inputs().
    Alternatively, the user can explicitly specify the path for the COLMAP
    ref images output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP ref images input file (default is None)

    Returns:
        (DataFrame) Dataframe containing camera position input data
    """
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either ref image data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'ref_images.txt'
        )
    df = pd.read_csv(
        path,
        header=None,
        delim_whitespace=True,
        names = ['image_path', 'x', 'y', 'z'],
        dtype={
            'image_path': 'string',
            'x': 'float',
            'y': 'float',
            'z': 'float',
        }
    )
    df['position_input'] = df.apply(
        lambda row: np.array([row['x'], row['y'], row['z']]),
        axis=1
    )
    df.set_index('image_path', inplace=True)
    df = df.reindex(columns=[
        'position_input'
    ])
    return df

def compare_colmap_calibration_to_existing(
    colmap_output_df,
    existing_calibration_time
):
    """
    Compares COLMAP output to existing calibration data in Honeycomb for the same cameras.

    The COLMAP output dataframe must contain the following columns: device_id,
    camera_matrix, distortion_coefficients, image_width, image_height,
    rotation_vector, and translation_vector. This is consistent with column
    naming conventions of fetch_colmap_output_data_local(). All other columns
    are ignored.

    Output is a dataframe with a row for each camera that has calibration data
    in the COLMAP output and in Honeycomb (at the specified time). Index is
    camera device ID and columns are orientation_difference_angle_radians,
    orientation_difference_angle_degrees, orientation_difference_direction,
    position_difference_distance, and position_difference_direction.


    Args:
        colmap_output_df (DataFrame): Dataframe containing COLMAP output
        existing_calibration_time (datetime): Existing calibration datetime to compare with

    Returns:
        (DataFrame) Dataframe containing comparison info
    """
    new_calibrations =(
        colmap_output_df
        .dropna(subset=['device_id'])
        .reindex(columns=[
            'device_id',
            'camera_matrix',
            'distortion_coefficients',
            'image_width',
            'image_height',
            'rotation_vector',
            'translation_vector'
        ])
        .set_index('device_id')
        .to_dict(orient='index')
    )
    device_ids = list(new_calibrations.keys())
    old_calibrations = honeycomb_io.fetch_camera_calibrations(
        camera_ids=device_ids,
        start=existing_calibration_time,
        end=existing_calibration_time
    )
    calibration_comparisons = camera_calibration.analyze.compare_calibrations(
        old_calibrations=old_calibrations,
        new_calibrations=new_calibrations
    )
    calibration_comparisons_df = pd.DataFrame.from_dict(
        calibration_comparisons,
        orient='index'
    )
    calibration_comparisons_df.index.name='camera_id'
    return calibration_comparisons_df

def write_colmap_output_honeycomb(
    colmap_output_df,
    calibration_start,
    coordinate_space_id,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    """
    Writes camera calibration and position data generated by COLMAP to Honeycomb.

    The dataframe with the COLMAP output data must contain one row for each
    camera and the following columns: device_id, image_width, image_height,
    camera_matrix, distortion_coefficients, rotation_vector, translation_vector,
    and position.

    Requires the Honeycomb coordinate space ID for the coordinate space in which
    the extrinsic calibration and position data is calculated. This coordinate
    space must already be created in Honeycomb.

    Returns a dictionary with three items, each a list of Honeycomb IDs for the
    newly created objects: intrinsic_calibration_ids, extrinsic_calibration_ids,
    and position_assignment_ids.

    Args:
        colmap_output_df (DataFrame): Dataframe containing the COLMAP output data
        calibration_start (datetime): The start time for the new Honeycomb objects
        coordinate_space_id (str): The Honeycomb coordinate space ID for extrinsic calibration and position assignment objects
        client (MinimalHoneycombClient): Client object to use when connecting with Honeycomb (default is None)
        uri (str): URI to use for Honeycomb client (default is None)
        token_uri (str): Token URI to use for Honeycomb client (default is None)
        audience (str): Audience to use for Honeycomb client (default is None)
        client_id (str): Client ID to use for Honeycomb client (default is None)
        client_secret (str): Client secret to use for Honeycomb client (default is None)

    Returns:
        (dict): Dictionary containing Honeycomb IDs for new objects
    """
    calibration_data_columns = [
        'device_id',
        'image_width',
        'image_height',
        'camera_matrix',
        'distortion_coefficients',
        'rotation_vector',
        'translation_vector',
        'position'
    ]
    if not set(calibration_data_columns).issubset(set(colmap_output_df.columns)):
        raise ValueError('COLMAP output data must contain the following columns: {}'.format(
            calibration_data_columns
        ))
    colmap_output_df = colmap_output_df.dropna(subset=['device_id'])
    calibration_start = pd.to_datetime(calibration_start, utc=True).to_pydatetime()
    intrinsic_calibration_ids = honeycomb_io.write_intrinsic_calibration_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    extrinsic_calibration_ids = honeycomb_io.write_extrinsic_calibration_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        coordinate_space_id=coordinate_space_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    position_assignment_ids = honeycomb_io.write_position_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        coordinate_space_id=coordinate_space_id,
        assigned_type='DEVICE',
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    honeycomb_ids = {
        'intrinsic_calibration_ids': intrinsic_calibration_ids,
        'extrinsic_calibraion_ids': extrinsic_calibration_ids,
        'position_assignment_ids': position_assignment_ids
    }
    return honeycomb_ids

def colmap_parameters_to_opencv_parameters(colmap_parameters, colmap_camera_model):
    if colmap_camera_model == 'SIMPLE_PINHOLE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = None
    elif colmap_camera_model == 'PINHOLE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = None
    elif colmap_camera_model == 'SIMPLE_RADIAL':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            0.0,
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'RADIAL':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            colmap_parameters[4],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'OPENCV':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7]
        ])
    elif colmap_camera_model == 'OPENCV_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            0.0,
            0.0,
            colmap_parameters[6],
            colmap_parameters[7],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'FULL_OPENCV':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7],
            colmap_parameters[8],
            colmap_parameters[9],
            colmap_parameters[10],
            colmap_parameters[11]
        ])
    elif colmap_camera_model == 'SIMPLE_RADIAL_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            0.0,
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'RADIAL_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            colmap_parameters[4],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'THIN_PRISM_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7],
            colmap_parameters[8],
            colmap_parameters[9],
            0.0,
            0.0,
            colmap_parameters[10],
            colmap_parameters[11],
            0.0,
            0.0
        ])
    else:
        raise ValueError('Camera model {} not found'.format(colmap_camera_model))
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    return camera_matrix, distortion_coefficients

def extract_colmap_image_calibration_data(
    input_path,
    output_path
):
    output_lines = list()
    with open(input_path, 'r') as fp:
        for line in fp.readlines():
            m = re.match(CALIBRATION_DATA_RE, line)
            if m:
                output_line = ','.join([
                    m.group('colmap_image_id'),
                    m.group('qw'),
                    m.group('qx'),
                    m.group('qy'),
                    m.group('qz'),
                    m.group('tx'),
                    m.group('ty'),
                    m.group('tz'),
                    m.group('colmap_camera_id'),
                    m.group('image_path')
                ])
                output_lines.append(output_line)
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(output_lines))
