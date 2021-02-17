import minimal_honeycomb
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def write_intrinsic_calibration_data(
    data,
    start_datetime,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    intrinsic_calibration_data_columns = [
        'device_id',
        'image_width',
        'image_height',
        'camera_matrix',
        'distortion_coefficients'
    ]
    if not set(intrinsic_calibration_data_columns).issubset(set(data.columns)):
        raise ValueError('Data must contain the following columns: {}'.format(
            intrinsic_calibration_data_columns
        ))
    intrinsic_calibration_data_df = data.reset_index().reindex(columns=intrinsic_calibration_data_columns)
    intrinsic_calibration_data_df.rename(columns={'device_id': 'device'}, inplace=True)
    intrinsic_calibration_data_df['start'] = minimal_honeycomb.to_honeycomb_datetime(start_datetime)
    intrinsic_calibration_data_df['camera_matrix'] = intrinsic_calibration_data_df['camera_matrix'].apply(lambda x: x.tolist())
    intrinsic_calibration_data_df['distortion_coefficients'] = intrinsic_calibration_data_df['distortion_coefficients'].apply(lambda x: x.tolist())
    records = intrinsic_calibration_data_df.to_dict(orient='records')
    if client is None:
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    result=client.bulk_mutation(
        request_name='createIntrinsicCalibration',
        arguments={
            'intrinsicCalibration': {
                'type': 'IntrinsicCalibrationInput',
                'value': records
            }
        },
        return_object=[
            'intrinsic_calibration_id'
        ]
    )
    ids = None
    if len(result) > 0:
        ids = [datum.get('intrinsic_calibration_id') for datum in result]
    return ids

def write_extrinsic_calibration_data(
    data,
    start_datetime,
    coordinate_space_id,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    extrinsic_calibration_data_columns = [
        'device_id',
        'rotation_vector',
        'translation_vector'
    ]
    if not set(extrinsic_calibration_data_columns).issubset(set(data.columns)):
        raise ValueError('Data must contain the following columns: {}'.format(
            extrinsic_calibration_data_columns
        ))
    extrinsic_calibration_data_df = data.reset_index().reindex(columns=extrinsic_calibration_data_columns)
    extrinsic_calibration_data_df.rename(columns={'device_id': 'device'}, inplace=True)
    extrinsic_calibration_data_df['start'] = minimal_honeycomb.to_honeycomb_datetime(start_datetime)
    extrinsic_calibration_data_df['coordinate_space'] = coordinate_space_id
    extrinsic_calibration_data_df['rotation_vector'] = extrinsic_calibration_data_df['rotation_vector'].apply(lambda x: x.tolist())
    extrinsic_calibration_data_df['translation_vector'] = extrinsic_calibration_data_df['translation_vector'].apply(lambda x: x.tolist())
    records = extrinsic_calibration_data_df.to_dict(orient='records')
    if client is None:
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    result=client.bulk_mutation(
        request_name='createExtrinsicCalibration',
        arguments={
            'extrinsicCalibration': {
                'type': 'ExtrinsicCalibrationInput',
                'value': records
            }
        },
        return_object=[
            'extrinsic_calibration_id'
        ]
    )
    ids = None
    if len(result) > 0:
        ids = [datum.get('extrinsic_calibration_id') for datum in result]
    return ids

def write_position_data(
    data,
    start_datetime,
    coordinate_space_id,
    assigned_type='DEVICE',
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    position_data_columns = [
        'device_id',
        'position'
    ]
    if not set(position_data_columns).issubset(set(data.columns)):
        raise ValueError('Data must contain the following columns: {}'.format(
            position_data_columns
        ))
    position_data_df = data.reset_index().reindex(columns=position_data_columns)
    position_data_df.rename(columns={'device_id': 'assigned'}, inplace=True)
    position_data_df.rename(columns={'position': 'coordinates'}, inplace=True)
    position_data_df['start'] = minimal_honeycomb.to_honeycomb_datetime(start_datetime)
    position_data_df['assigned_type'] = assigned_type
    position_data_df['coordinate_space'] = coordinate_space_id
    position_data_df['coordinates'] = position_data_df['coordinates'].apply(lambda x: x.tolist())
    records = position_data_df.to_dict(orient='records')
    if client is None:
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    result=client.bulk_mutation(
        request_name='assignToPosition',
        arguments={
            'positionAssignment': {
                'type': 'PositionAssignmentInput!',
                'value': records
            }
        },
        return_object=[
            'position_assignment_id'
        ]
    )
    ids = None
    if len(result) > 0:
        ids = [datum.get('position_assignment_id') for datum in result]
    return ids

def fetch_environment_id(
    environment_name,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    result = client.bulk_query(
        request_name='findEnvironments',
        arguments={
            'name': {
                'type': 'String',
                'value': environment_name
            }
        },
        return_data=[
            'environment_id'
        ],
        id_field_name='environment_id'
    )
    if len(result) == 0:
        raise ValueError('Environment {} not found'.format(environment_name))
    if len(result) > 1:
        raise ValueError('More than one environment found with name {}'.format(environment_name))
    environment_id = result[0].get('environment_id')
    return environment_id

def fetch_device_positions(
    environment_id,
    datetime,
    device_types,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    result=client.bulk_query(
        request_name='searchAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'operator': 'AND',
                    'children': [
                        {
                            'field': 'environment',
                            'operator': 'EQ',
                            'value': environment_id
                        },
                        {
                            'field': 'assigned_type',
                            'operator': 'EQ',
                            'value': 'DEVICE'
                        }
                    ]
                }
            }
        },
        return_data=[
            'assignment_id',
            'start',
            'end',
            {'assigned': [
                {'... on Device': [
                    'device_id',
                    'name',
                    'device_type',
                    {'position_assignments': [
                        'start',
                        'end',
                        {'coordinate_space': [
                            'space_id'
                        ]},
                        'coordinates'
                    ]}
                ]}
            ]}
        ],
        id_field_name='assignment_id'
    )
    logger.info('Fetched {} device assignments'.format(len(result)))
    device_assignments = minimal_honeycomb.filter_assignments(
        result,
        start_time=datetime,
        end_time=datetime
    )
    logger.info('{} of these device assignments are active at specified datetime'.format(len(device_assignments)))
    device_assignments = list(filter(lambda x: x.get('assigned', {}).get('device_type') in device_types, result))
    logger.info('{} of these device assignments correspond to target device types'.format(len(device_assignments)))
    device_positions = dict()
    for device_assignment in device_assignments:
        device_id = device_assignment.get('assigned', {}).get('device_id')
        device_name = device_assignment.get('assigned', {}).get('name')
        if device_name is None:
            logger.info('Device {} has no name. Skipping.'.format(device_id))
            continue
        position_assignments = device_assignment.get('assigned', {}).get('position_assignments')
        if position_assignments is None:
            continue
        logger.info('Device {} has {} position assignments'.format(device_name, len(position_assignments)))
        position_assignments = minimal_honeycomb.filter_assignments(
            position_assignments,
            start_time=datetime,
            end_time=datetime
        )
        if len(position_assignments) > 1:
            raise ValueError('Device {} has multiple position assignments at specified datetime'.format(device_name))
        if len(position_assignments) == 0:
            logger.info('Device {} has no position assignments at specified datetime'.format(device_name))
            continue
        position_assignment = position_assignments[0]
        device_positions[device_id] = {
            'name': device_name,
            'position': position_assignment.get('coordinates')
        }
    return device_positions

def fetch_device_position(
    device_id,
    datetime,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    result=client.bulk_query(
        request_name='searchPositionAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'operator': 'AND',
                    'children': [
                        {
                            'field': 'assigned',
                            'operator': 'EQ',
                            'value': device_id
                        },
                        {
                            'field': 'start',
                            'operator': 'LTE',
                            'value': minimal_honeycomb.to_honeycomb_datetime(datetime)
                        },
                        {
                            'operator': 'OR',
                            'children': [
                                {
                                    'field': 'end',
                                    'operator': 'GTE',
                                    'value': minimal_honeycomb.to_honeycomb_datetime(datetime)
                                },
                                {
                                    'field': 'end',
                                    'operator': 'ISNULL'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        return_data=[
            'position_assignment_id',
            'coordinates'
        ],
        id_field_name='position_assignment_id'
    )
    if len(result) == 0:
        return None
    if len(result) > 1:
        raise ValueError('More than one position assignment consistent with specified device ID and time')
    device_position = result[0].get('coordinates')
    return device_position

def fetch_assignment_id_lookup(
    assignment_ids,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    assignment_ids = list(assignment_ids)
    if client is None:
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    result = client.bulk_query(
        request_name='searchAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'assignment_id',
                    'operator': 'IN',
                    'values': assignment_ids
                }
            }
        },
        return_data = [
            'assignment_id',
            {'assigned': [
                {'... on Device': [
                    'device_id'
                ]},
                {'... on Person': [
                    'person_id'
                ]},
                {'... on Material': [
                    'material_id'
                ]},
                {'... on Tray': [
                    'tray_id'
                ]},
            ]}
        ],
        id_field_name='assignment_id'
    )
    if len(result) == 0:
        return None
    records = list()
    for datum in result:
        records.append({
        'assignment_id': datum.get('assignment_id'),
        'device_id': datum.get('assigned').get('device_id'),
        'person_id': datum.get('assigned').get('person_id'),
        'material_id': datum.get('assigned').get('material_id'),
        'tray_id': datum.get('assigned').get('tray_id')
        })
    df = pd.DataFrame.from_records(records)
    df.set_index('assignment_id', inplace=True)
    return df


def extract_honeycomb_id(string):
    id = None
    m = re.search(
        '(?P<id>[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12})',
        string
    )
    if m:
        id = m.group('id')
    return id

def fetch_camera_names(
    camera_ids,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Fetching camera names for specified camera device IDs')
    result = client.bulk_query(
        request_name='searchDevices',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device_id',
                    'operator': 'IN',
                    'values': camera_ids
                }
            }
        },
        return_data=[
            'device_id',
            'name'
        ],
        id_field_name = 'device_id',
        chunk_size=chunk_size
    )
    camera_names = {device.get('device_id'): device.get('name') for device in result}
    logger.info('Fetched {} camera names'.format(len(camera_names)))
    return camera_names

def fetch_camera_calibrations(
    camera_ids,
    start=None,
    end=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    intrinsic_calibrations = fetch_intrinsic_calibrations(
        camera_ids=camera_ids,
        start=start,
        end=end,
        chunk_size=chunk_size,
        client=None,
        uri=None,
        token_uri=None,
        audience=None,
        client_id=None,
        client_secret=None
    )
    extrinsic_calibrations = fetch_extrinsic_calibrations(
        camera_ids=camera_ids,
        start=start,
        end=end,
        chunk_size=chunk_size,
        client=None,
        uri=None,
        token_uri=None,
        audience=None,
        client_id=None,
        client_secret=None
    )
    camera_calibrations = dict()
    for camera_id in camera_ids:
        if camera_id not in intrinsic_calibrations.keys():
            logger.warning('No intrinsic calibration found for camera ID {}'.format(
                camera_id
            ))
            continue
        if camera_id not in extrinsic_calibrations.keys():
            logger.warning('No extrinsic calibration found for camera ID {}'.format(
                camera_id
            ))
            continue
        camera_calibrations[camera_id] = {**intrinsic_calibrations[camera_id], **extrinsic_calibrations[camera_id]}
    return camera_calibrations

def fetch_intrinsic_calibrations(
    camera_ids,
    start=None,
    end=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Fetching intrinsic calibrations for specified camera device IDs and time span')
    result = client.bulk_query(
        request_name='searchIntrinsicCalibrations',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device',
                    'operator': 'IN',
                    'values': camera_ids
                }
            }
        },
        return_data=[
            'intrinsic_calibration_id',
            'start',
            'end',
            {'device': [
                'device_id'
            ]},
            'camera_matrix',
            'distortion_coefficients',
            'image_width',
            'image_height'
        ],
        id_field_name = 'intrinsic_calibration_id',
        chunk_size=chunk_size
    )
    logger.info('Fetched {} intrinsic calibrations for specified camera IDs'.format(len(result)))
    filtered_result = minimal_honeycomb.filter_assignments(
        result,
        start,
        end
    )
    logger.info('{} intrinsic calibrations are consistent with specified start and end times'.format(len(filtered_result)))
    intrinsic_calibrations = dict()
    for datum in filtered_result:
        camera_id = datum.get('device').get('device_id')
        if camera_id in intrinsic_calibrations.keys():
            raise ValueError('More than one intrinsic calibration found for camera {}'.format(
                camera_id
            ))
        intrinsic_calibrations[camera_id] = {
            'camera_matrix': np.asarray(datum.get('camera_matrix')),
            'distortion_coefficients': np.asarray(datum.get('distortion_coefficients')),
            'image_width': datum.get('image_width'),
            'image_height': datum.get('image_height')
        }
    return intrinsic_calibrations

def fetch_extrinsic_calibrations(
    camera_ids,
    start=None,
    end=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Fetching extrinsic calibrations for specified camera device IDs and time span')
    result = client.bulk_query(
        request_name='searchExtrinsicCalibrations',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device',
                    'operator': 'IN',
                    'values': camera_ids
                }
            }
        },
        return_data=[
            'extrinsic_calibration_id',
            'start',
            'end',
            {'device': [
                'device_id'
            ]},
            {'coordinate_space': [
                'space_id'
            ]},
            'translation_vector',
            'rotation_vector'
        ],
        id_field_name = 'extrinsic_calibration_id',
        chunk_size=chunk_size
    )
    logger.info('Fetched {} extrinsic calibrations for specified camera IDs'.format(len(result)))
    filtered_result = minimal_honeycomb.filter_assignments(
        result,
        start,
        end
    )
    logger.info('{} extrinsic calibrations are consistent with specified start and end times'.format(len(filtered_result)))
    extrinsic_calibrations = dict()
    space_ids = list()
    for datum in filtered_result:
        camera_id = datum.get('device').get('device_id')
        space_id = datum.get('coordinate_space').get('space_id')
        space_ids.append(space_id)
        if camera_id in extrinsic_calibrations.keys():
            raise ValueError('More than one extrinsic calibration found for camera {}'.format(
                camera_id
            ))
        extrinsic_calibrations[camera_id] = {
            'space_id': space_id,
            'rotation_vector': np.asarray(datum.get('rotation_vector')),
            'translation_vector': np.asarray(datum.get('translation_vector'))
        }
    if len(np.unique(space_ids)) > 1:
        raise ValueError('More than one coordinate space found among fetched calibrations')
    return extrinsic_calibrations

def fetch_camera_device_id_lookup(
    assignment_ids,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = generate_client(
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    result = client.bulk_query(
        request_name='searchAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'assignment_id',
                    'operator': 'IN',
                    'values': assignment_ids
                }
        }},
        return_data=[
            'assignment_id',
            {'assigned': [
                {'... on Device': [
                    'device_id'
                ]}
            ]}
        ],
        id_field_name='assignment_id'
    )
    camera_device_id_lookup = dict()
    for datum in result:
        camera_device_id_lookup[datum.get('assignment_id')] = datum.get('assigned').get('device_id')
    return camera_device_id_lookup


def generate_client(
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if client is None:
        client=minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    return client
