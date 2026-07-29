'''Safe output-directory resolution for Auto runs.

The resolver is deliberately pure with respect to the controller and hardware:
it chooses a non-conflicting relative output name but never creates a folder or
contacts an external service.  The controller creates the approved directory
only after this check succeeds.
'''

import os


class AutoOutputDirectoryConflictError(RuntimeError):
    '''Raised when an Auto output-folder collision is not explicitly approved.'''


def _validate_relative_output_name(data_dir):
    '''Returns a safe relative output path without changing the filesystem.'''
    if data_dir is None:
        raise AutoOutputDirectoryConflictError(
            'Header data_dir must be a nonempty relative path below '
            'Protocol_Outputs.'
        )

    normalized_data_dir = os.path.normpath(str(data_dir).strip())

    if (
        not normalized_data_dir
        or normalized_data_dir == '.'
        or os.path.isabs(normalized_data_dir)
        or normalized_data_dir == os.pardir
        or normalized_data_dir.startswith(os.pardir + os.sep)
    ):
        raise AutoOutputDirectoryConflictError(
            'Header data_dir must be a nonempty relative path below '
            'Protocol_Outputs.'
        )

    return normalized_data_dir


def _first_available_relative_name(
    output_root,
    requested_relative_name,
    path_exists
):
    '''Returns the first ``_N`` sibling that does not already exist.'''
    parent_name, base_name = os.path.split(requested_relative_name)
    suffix = 1

    while True:
        candidate_base_name = '{}_{}'.format(base_name, suffix)
        candidate_relative_name = os.path.join(
            parent_name,
            candidate_base_name
        ) if parent_name else candidate_base_name
        candidate_path = os.path.join(output_root, candidate_relative_name)
        if not path_exists(candidate_path):
            return candidate_relative_name
        suffix += 1


def resolve_auto_output_directory(
    output_root,
    data_dir,
    input_func,
    interactive,
    path_exists=os.path.exists
):
    '''Resolves one explicitly approved, unused Auto output directory.

    A new requested path is returned unchanged. If the requested directory
    already exists, the caller must be interactive and explicitly approve the
    first unused ``_N`` sibling. This function never creates directories and
    never silently redirects output.
    '''
    output_root = os.path.abspath(os.fspath(output_root))
    requested_relative_name = _validate_relative_output_name(data_dir)
    requested_path = os.path.abspath(
        os.path.join(output_root, requested_relative_name)
    )

    if os.path.commonpath([output_root, requested_path]) != output_root:
        raise AutoOutputDirectoryConflictError(
            'Header data_dir must resolve below Protocol_Outputs.'
        )

    if not path_exists(requested_path):
        return {
            'requested_data_dir': requested_relative_name,
            'effective_data_dir': requested_relative_name,
            'output_path': requested_path,
            'was_renamed_for_collision': False
        }

    proposed_relative_name = _first_available_relative_name(
        output_root,
        requested_relative_name,
        path_exists
    )
    proposed_path = os.path.join(output_root, proposed_relative_name)

    if not interactive:
        raise AutoOutputDirectoryConflictError(
            'Requested Auto output directory already exists: {}. '
            'Start from an interactive terminal and explicitly approve {} '
            'or choose a new Header data_dir.'.format(
                requested_path,
                proposed_path
            )
        )

    response = str(input_func(
        '\n<<controller>> requested Auto output directory already exists:\n'
        '{}\n'
        '<<controller>> proposed new output directory:\n'
        '{}\n'
        '<<controller>> type yes to use the proposed new directory, or '
        'anything else to stop: '.format(requested_path, proposed_path)
    )).strip().lower()

    if response != 'yes':
        raise AutoOutputDirectoryConflictError(
            'Auto output-directory collision was not approved. No output '
            'directory was created.'
        )

    return {
        'requested_data_dir': requested_relative_name,
        'effective_data_dir': proposed_relative_name,
        'output_path': proposed_path,
        'was_renamed_for_collision': True
    }
