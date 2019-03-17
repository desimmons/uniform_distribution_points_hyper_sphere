# uniform_distribution_points_hyper_sphere
The repository provides an implementation of the algorithm presented in the paper [Uniform distribution of points on a hyper-sphere with applications to vector bit-plane encoding](https://www.researchgate.net/publication/3359197_Uniform_distribution_of_points_on_a_hyper-sphere_with_applicationsto_vector_bit-plane_encoding). The algorithm produces an approximately uuniform set of unit vectors distributed over a unit hypersphere. The algorithm becomes exact in the limit as the nuber of embedded vectors goes to infinity. The algorithm will sometimes struggle to converge (this is particularly an issue when the number of embedded vectors ~ the dimension of the embedding).

## How to run
``` 
conda create --name <insert_env_name> --file req.txt
conda activate <insert_env_name>
python construct_matrix.py --dim 3 --num_vecs 30
```
