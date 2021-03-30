PARAMS = {
    'gcn-align':
        {'encoder': 'gcn-align',
         'decoder': ['align'],
         'hiddens': [100, 100, 100],
         'sampling': ['N'],
         'margin': [1],
         'alpha': [1],
         'feat_drop': 0.,
         'lr': 0.005,
         'dist': 'euclidean'
         },
    'mtranse':
        {'encoder': None,
         'decoder': ['transe', 'mtranse_align'],
         'hiddens': [100],
         'sampling': ['.', '.'],
         'margin': [0, 0],
         'alpha': [1, 50],
         'feat_drop': 0.,
         'k': [0, 0],
         'lr': 0.01,
         'dist': 'euclidean'
         },
    'rotate': dict(
        encoder=None,
        decoder=['rotate'],
        sampling=['T'],
        k=[5],
        margin=[5],
        hiddens=[200],
        alpha=[1],
        update=50,
        lr=0.05,
        dist='euclidean',
        share=True
    ),
    'kecg': dict(
        encoder='kecg',
        decoder=['transe', 'align'],
        sampling=['T', 'N'],
        k=[5, 25],
        margin=[3, 3],
        hiddens=[100, 100, 100],
        alpha=[1, 1],
        heads=[2, 2],
        lr=0.005,
        dist='manhattan'
    )

}
