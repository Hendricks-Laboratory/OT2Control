import ot2lib 


sbs = ot2lib.get_side_by_side_df(use_cache=True)
sbs['flag'] = sbs.apply(lambda row: ot2lib.is_valid_sbs(row), axis=1)
filtered_sbs = sbs.loc[~sbs['flag']]
if filtered_sbs.empty:
    print('congrats! Volumes and deck_poses look good!')
else:
    print('errors on following columns')
    print(filtered_sbs)
