#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ctime>

// Wishlist:
// More utility transforms on mat2
// Grid overlay for easy camera repositioning
// template mat2 so that we can switch between single and double precision on the fly.
// Zoom tweening

#ifdef __CUDACC__
void handle_cuda_err(cudaError_t err, int line) {
	if (err != cudaSuccess) {
		printf("%s on %d:\n%s\n", cudaGetErrorName(err), line, cudaGetErrorString(err));
	}
}
#endif

// Transformation matrix 2D
class mat2 {
public:
	float x[2];
	float y[2];
	float o[2];
	
	float zoom; // TODO: Replace with stateless getter.
	
	// From offset and scale
	mat2(float offset_x, float offset_y, float scale_x, float scale_y) : o{offset_x, offset_y}, x{scale_x, 0}, y{0, scale_y} {}
	
	// From camera center, and viewport size, and zoom
	// A zoom of 1 renders a region 1 unit wide of the complex plane. Height matches the aspect ratio of the viewport.
	mat2(float cx, float cy, int width, int height, float zoom) : o{cx - 0.5f/zoom, cy - 0.5f/zoom * height/width}, x{1/zoom/width, 0}, y{0, 1/zoom/width}, zoom(zoom) {}
	
	__host__ __device__ 
	void transform(float in[2]) {
		float in0_t = in[0];
		in[0] = in0_t * x[0] + in[1] * y[0] + o[0];
		in[1] = in0_t * x[1] + in[1] * y[1] + o[1];
	}
	
	void debug() {
		printf("x: %5.2f, %5.2f\ny: %5.2f, %5.2f\no: %5.2f, %5.2f\n", x[0], x[1], y[0], y[1], o[0], o[1]);
	}
};

// Matrix
class matrix {
public:
	int num_cols;
	int num_rows;
	
	float** data;
	
	matrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {
		data = new float*[num_rows];
		for (int r = 0; r < num_rows; r++) {
			data[r] = new float[num_cols];
		}
	}
	
	void fill(float value) {
		for (int r = 0; r < num_rows; r++) {
			for (int c = 0; c < num_cols; c++) {
				data[r][c] = value;
			}
		}
	}
	
	void swap_rows(int r1, int r2) {
		if (r1 < 0 || r1 >= num_rows || r2 < 0 || r2 >= num_rows) {
			printf("Attempting to swap rows %d and %d in a table with %d rows.\n", r1, r2, num_rows);
		}
		
		float* t = data[r1];
		data[r1] = data[r2];
		data[r2] = t;
	}
	
	// Add a scalar multiple of row r1 to row r2.
	void addscamul_row(int r1, int r2, float mul) {
		if (r1 < 0 || r1 >= num_rows || r2 < 0 || r2 >= num_rows) {
			printf("Attempting to addscamul rows %d and %d in a table with %d rows.\n", r1, r2, num_rows);
		}
		
		for (int c = 0; c < num_cols; c++) {
			data[r2][c] += data[r1][c] * mul;
		}
	}
	
	void divide_row(int r, float dividend) {
		if (r < 0 || r >= num_rows) {
			printf("Attempting to divide row %d in a table with %d rows.\n", r, num_rows);
		}
		if (dividend == 0) {
			printf("Attempting to divide row %d by 0.\n", r);
		}
		
		for (int c = 0; c < num_cols; c++) {
			data[r][c] /= dividend;
		}
	}
	
	void debug() {
		for (int r = 0; r < num_rows; r++) {
			for (int c = 0; c < num_cols; c++) {
				printf("%5.2f ", data[r][c]);
			}
			printf("\n");
		}
	}
	
	~matrix() {
		for (int r = 0; r < num_rows; r++) {
			delete[] data[r];
		}
		delete[] data;
	}
};

class color {
public:
	uint8_t r;
	uint8_t g;
	uint8_t b;
	
	__host__ __device__ 
	color() {}
	
	__host__ __device__ 
	color(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
	
	__host__ __device__ 
	static color lerp(const color& a, const color& b, float t) {
		return color(
			(b.r - a.r)*t + a.r,
			(b.g - a.g)*t + a.g,
			(b.b - a.b)*t + a.b
		);
	}
};

class cubic {
public:
	// y = ax^3 + bx^2 + cx + d
	float a;
	float b;
	float c;
	float d;
	
	cubic() {}
	
	// Create cubic function.
	cubic(float a, float b, float c, float d) : a(a), b(b), c(c), d(d) {}
	
	// Create linear function.
	cubic(float c, float d) : a(0), b(0), c(c), d(d) {}
	
	__host__ __device__
	float evaluate(float x) {
		float ret = a*(x*x*x) + b*(x*x) + c*x + d;
		//printf("%.4f * %.4f + %.4f * %.4f + %.4f * %.4f + %.4f -> %.2f\n", a, x*x*x, b, x*x, c, x, d, ret);
		return ret;
	}
	
	void debug() {
		printf("%.2fx^3 + %.2fx^2 + %.2fx + %.2f\n", a, b, c, d);
	}
};

class palette {
public:
	int max_colors;
	int num_colors; // The first color (passed to the constructor) is duplicated at the end, so positions[num_colors] and colors[num_colors] are defined.
	
	float* positions;
	color* colors;
	
	cubic* r_cubics;
	cubic* g_cubics;
	cubic* b_cubics;
	
	#ifdef __CUDACC__
	palette* this_g; // Pointer to this object on the GPU.
	#endif
	
	bool has_been_destroyed;
	
	palette(int max_colors, float pos, uint8_t r, uint8_t g, uint8_t b) : max_colors(max_colors), num_colors(1), has_been_destroyed(false), this_g(nullptr) {
		#ifdef __CUDACC__
		cudaError_t err = cudaGetLastError();
		handle_cuda_err(err, __LINE__);
		#endif
		
		#ifdef __CUDACC__
		err = cudaMallocManaged(&positions, (max_colors+1) * sizeof(float));
		handle_cuda_err(err, __LINE__);
		
		err = cudaMallocManaged(&colors, (max_colors+1) * sizeof(color));
		handle_cuda_err(err, __LINE__);
		
		err = cudaMallocManaged(&r_cubics, max_colors * sizeof(cubic));
		handle_cuda_err(err, __LINE__);
		
		err = cudaMallocManaged(&g_cubics, max_colors * sizeof(cubic));
		handle_cuda_err(err, __LINE__);
		
		err = cudaMallocManaged(&b_cubics, max_colors * sizeof(cubic));
		handle_cuda_err(err, __LINE__);
		#else
		positions = new float[max_colors+1];
		colors = new color[max_colors+1];
		
		r_cubics = new cubic[max_colors];
		g_cubics = new cubic[max_colors];
		b_cubics = new cubic[max_colors];
		#endif
		
		positions[0] = pos;
		colors[0] = color(r, g, b);
		
		positions[max_colors] = pos + 1;
		colors[max_colors] = color(r, g, b);
	}
	
	#ifdef __CUDACC__
	// Allocates space for this object on the GPU (if it does not already exist), copies this object there, and then returns the pointer.
	palette* get_gpu_ptr() {
		cudaError_t err = cudaGetLastError();
		handle_cuda_err(err, __LINE__);
		
		if (this_g == nullptr) {
			err = cudaMalloc(&this_g, sizeof(palette));
			handle_cuda_err(err, __LINE__);
		}
		
		err = cudaMemcpy(this_g, this, sizeof(palette), cudaMemcpyHostToDevice);
		handle_cuda_err(err, __LINE__);
		
		return this_g;
	}
	#endif
	
	void add_color(float pos, uint8_t r, uint8_t g, uint8_t b) {
		if (num_colors == max_colors) {
			printf("Too many colors!\n");
			return;
		}
		
		if (pos < 0 || pos > 1) {
			printf("pos must be in the range [0, 1]\n");
			return;
		}
		
		positions[num_colors] = pos;
		colors[num_colors] = color(r, g, b);
		num_colors++;
	}
	
	// Cache linear interpolation of colors.
	void cache_cubics_as_linear() {
		if (num_colors < max_colors) {
			printf("Not enough colors! (%d/%d)\n", num_colors, max_colors);
			return;
		}
		
		for (int i = 0; i < num_colors; i++) {
			int n_i = (i + 1);
			
			// Create interpolation cubics for linear interpolation.
			r_cubics[i] = cubic( (colors[n_i].r - colors[i].r) / (positions[n_i] - positions[i]), colors[i].r );
			g_cubics[i] = cubic( (colors[n_i].g - colors[i].g) / (positions[n_i] - positions[i]), colors[i].g );
			b_cubics[i] = cubic( (colors[n_i].b - colors[i].b) / (positions[n_i] - positions[i]), colors[i].b );
		}
	}
	
	// Cache cubic interpolation of colors; evaluation may exceed 255 or dip below 0.
	void cache_cubics() {
		if (num_colors < max_colors) {
			printf("Not enough colors! (%d/%d)\n", num_colors, max_colors);
			return;
		}
		
		int last_column = num_colors*4; // Index of the right-most column in the matrix; also the number of rows;
		
		// Create matrices and fill with 0s
		matrix r(last_column, last_column + 1);
		matrix g(last_column, last_column + 1);
		matrix b(last_column, last_column + 1);
		r.fill(0); g.fill(0); b.fill(0);
		
		// Load matrices
		int row = 0; // next row to write an equation into.
		
		float x, ry, gy, by;
		// Load first and last equivalences.
		for (int i = 0; i < num_colors; i++) {
			x = positions[i];
			ry = colors[i].r;
			gy = colors[i].g;
			by = colors[i].b;
			
			r.data[row][i*4  ] = x*x*x;
			r.data[row][i*4+1] = x*x;
			r.data[row][i*4+2] = x;
			r.data[row][i*4+3] = 1;
			
			r.data[row][last_column] = ry;
			
			g.data[row][i*4  ] = x*x*x;
			g.data[row][i*4+1] = x*x;
			g.data[row][i*4+2] = x;
			g.data[row][i*4+3] = 1;
			
			g.data[row][last_column] = gy;
			
			b.data[row][i*4  ] = x*x*x;
			b.data[row][i*4+1] = x*x;
			b.data[row][i*4+2] = x;
			b.data[row][i*4+3] = 1;
			
			b.data[row][last_column] = by;
			
			row++;
			
			x = positions[i+1];
			ry = colors[i+1].r;
			gy = colors[i+1].g;
			by = colors[i+1].b;
			
			r.data[row][i*4  ] = x*x*x;
			r.data[row][i*4+1] = x*x;
			r.data[row][i*4+2] = x;
			r.data[row][i*4+3] = 1;
			
			r.data[row][last_column] = ry;
			
			g.data[row][i*4  ] = x*x*x;
			g.data[row][i*4+1] = x*x;
			g.data[row][i*4+2] = x;
			g.data[row][i*4+3] = 1;
			
			g.data[row][last_column] = gy;
			
			b.data[row][i*4  ] = x*x*x;
			b.data[row][i*4+1] = x*x;
			b.data[row][i*4+2] = x;
			b.data[row][i*4+3] = 1;
			
			b.data[row][last_column] = by;
			
			row++;
		}
		
		// Load first-derivative equivalencies
		for (int i = 0; i < num_colors-1; i++) {
			x = positions[i+1];
			
			r.data[row][i*4  ] = 3 * x*x;
			r.data[row][i*4+1] = 2 * x;
			r.data[row][i*4+2] = 1;
			
			r.data[row][(i+1)*4  ] = -3 * x*x;
			r.data[row][(i+1)*4+1] = -2 * x;
			r.data[row][(i+1)*4+2] = -1;
			
			g.data[row][i*4  ] = 3 * x*x;
			g.data[row][i*4+1] = 2 * x;
			g.data[row][i*4+2] = 1;
			
			g.data[row][(i+1)*4  ] = -3 * x*x;
			g.data[row][(i+1)*4+1] = -2 * x;
			g.data[row][(i+1)*4+2] = -1;
			
			b.data[row][i*4  ] = 3 * x*x;
			b.data[row][i*4+1] = 2 * x;
			b.data[row][i*4+2] = 1;
			
			b.data[row][(i+1)*4  ] = -3 * x*x;
			b.data[row][(i+1)*4+1] = -2 * x;
			b.data[row][(i+1)*4+2] = -1;
			
			row++;
		}
		
		// Load second-derivative equivalencies
		for (int i = 0; i < num_colors-1; i++) {
			x = positions[i+1];
			
			r.data[row][i*4  ] = 6 * x;
			r.data[row][i*4+1] = 2;
			
			r.data[row][(i+1)*4  ] = -6 * x;
			r.data[row][(i+1)*4+1] = -2;
			
			g.data[row][i*4  ] = 6 * x;
			g.data[row][i*4+1] = 2;
			
			g.data[row][(i+1)*4  ] = -6 * x;
			g.data[row][(i+1)*4+1] = -2;
			
			b.data[row][i*4  ] = 6 * x;
			b.data[row][i*4+1] = 2;
			
			b.data[row][(i+1)*4  ] = -6 * x;
			b.data[row][(i+1)*4+1] = -2;
			
			row++;
		}
		
		// First and last x values.
		float x_s = positions[0];
		float x_e = positions[num_colors];
		
		// Load end conditions
		r.data[row][0] = 3 * x_s*x_s;
		r.data[row][1] = 2 * x_s;
		r.data[row][2] = 1;
		
		r.data[row][(num_colors-1)*4  ] = -3 * x_e*x_e;
		r.data[row][(num_colors-1)*4+1] = -2 * x_e;
		r.data[row][(num_colors-1)*4+2] = -1;
		
		g.data[row][0] = 3 * x_s*x_s;
		g.data[row][1] = 2 * x_s;
		g.data[row][2] = 1;
		
		g.data[row][(num_colors-1)*4  ] = -3 * x_e*x_e;
		g.data[row][(num_colors-1)*4+1] = -2 * x_e;
		g.data[row][(num_colors-1)*4+2] = -1;
		
		b.data[row][0] = 3 * x_s*x_s;
		b.data[row][1] = 2 * x_s;
		b.data[row][2] = 1;
		
		b.data[row][(num_colors-1)*4  ] = -3 * x_e*x_e;
		b.data[row][(num_colors-1)*4+1] = -2 * x_e;
		b.data[row][(num_colors-1)*4+2] = -1;
		
		row++;
		
		r.data[row][0] = 6 * x_s;
		r.data[row][1] = 2;
		
		r.data[row][(num_colors-1)*4  ] = -6 * x_e;
		r.data[row][(num_colors-1)*4+1] = -2;
		
		g.data[row][0] = 6 * x_s;
		g.data[row][1] = 2;
		
		g.data[row][(num_colors-1)*4  ] = -6 * x_e;
		g.data[row][(num_colors-1)*4+1] = -2;
		
		b.data[row][0] = 6 * x_s;
		b.data[row][1] = 2;
		
		b.data[row][(num_colors-1)*4  ] = -6 * x_e;
		b.data[row][(num_colors-1)*4+1] = -2;
		
		// Solve the loaded system of equations.
		
		// Convert to row echelon form.
		matrix* each_mat[3] = {&r, &g, &b};
		for (int m_i = 0; m_i < 3; m_i++) {
			matrix& m = *each_mat[m_i];
			
			for (int k = 0; k < last_column; k++) {
				// Get the largest i value. Swap the row containing that value with this row.
				float max_i = 0;
				int max_i_row = -1;
				for (int i = k; i < last_column; i++) {
					if (abs(m.data[i][k]) > max_i) {
						max_i = abs(m.data[i][k]);
						max_i_row = i;
					}
				}
				m.swap_rows(k, max_i_row);
				
				if (max_i == 0) {
					printf("Unable to solve system of equations.\n");
				}
				
				// Zero all elements beneath k
				for (int i = k+1; i < last_column; i++) {
					if (m.data[k][k] != 0) {
						float mul = -m.data[i][k] / m.data[k][k];
						
						m.addscamul_row(k, i, mul);
					}
				}
			}
		
			// Back-substitute from the bottom up.
			for (int i = last_column-1; i >= 0; i--) {
				for (int j = last_column-1; j > i; j--) {
					m.data[i][last_column] -= m.data[j][last_column] * m.data[i][j];
				}
				
				m.data[i][last_column] /= m.data[i][i];
			}
		}
		
		// Write the results into the cubics.
		for (int i = 0; i < num_colors; i++) {
			r_cubics[i] = cubic(
				r.data[i*4  ][last_column],
				r.data[i*4+1][last_column],
				r.data[i*4+2][last_column],
				r.data[i*4+3][last_column]
			);
			
			g_cubics[i] = cubic(
				g.data[i*4  ][last_column],
				g.data[i*4+1][last_column],
				g.data[i*4+2][last_column],
				g.data[i*4+3][last_column]
			);
			
			b_cubics[i] = cubic(
				b.data[i*4  ][last_column],
				b.data[i*4+1][last_column],
				b.data[i*4+2][last_column],
				b.data[i*4+3][last_column]
			);
		}
		
		// r_cubics[1].debug();
		// printf("%.2f -> %.2f\n", positions[1], positions[2]);
		// printf("%u\n", evaluate(0.2).r);
		
		// printf("Red:\n");
		// r.debug();
		
		// printf("Green:\n");
		// g.debug();
		
		// printf("Blue:\n");
		// b.debug();
	}
	
	__host__ __device__ color evaluate(float x) {
		if (x < 0 || x > 1) {
			printf("x must be in the range 0 to 1.\n");
			return color(255, 0, 140);
		}
		//x = fmodf(x, 1);
		
		for (int i = 0; i <= num_colors; i++) {
			if (positions[i] > x) {
				if (i == 0) {
					i = num_colors;
					x += 1;
				}
				
				// printf("Evaluating x = %.2f before %.2f on cubic %d.\n", x, positions[i], i-1);
				
				return color(
					min(max(r_cubics[i-1].evaluate(x), (float) 0), (float) 255),
					min(max(g_cubics[i-1].evaluate(x), (float) 0), (float) 255),
					min(max(b_cubics[i-1].evaluate(x), (float) 0), (float) 255)
				);
			}
		}
		
		printf("Palette evaluation failed.\n");
		return color(255, 0, 140);
	}
	
	void debug(const char* fn) {
		int width = 600;
		int height = 80;
		
		uint8_t* img = new uint8_t[width*height*3];
		
		for (int x = 0; x < width; x++) {
			color col = evaluate((float) x / width);
			
			for (int y = 0; y < height; y++) {
				color ncol = col;
				if ((int) (col.r * height / 255) == height - y) {ncol.r = 255; ncol.g = 0; ncol.b = 0;}
				if ((int) (col.g * height / 255) == height - y) {ncol.r = 0; ncol.g = 255; ncol.b = 0;}
				if ((int) (col.b * height / 255) == height - y) {ncol.r = 0; ncol.g = 0; ncol.b = 255;}
				
				int ind = (y*width + x)*3;
				img[ind  ] = ncol.r;
				img[ind+1] = ncol.g;
				img[ind+2] = ncol.b;
			}
		}
		
		char hdr[32];
		int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
		FILE* fout = fopen(fn, "wb");
		fwrite(hdr, 1, hdr_len, fout);
		fwrite(img, 1, width*height*3, fout);
		fclose(fout);
	}
	
	~palette() {
		#ifdef __CUDACC__
		cudaError_t err = cudaGetLastError();
		handle_cuda_err(err, __LINE__);
		#endif
		
		#ifdef __CUDACC__
		err = cudaFree(positions);
		handle_cuda_err(err, __LINE__);
		
		err = cudaFree(colors);
		handle_cuda_err(err, __LINE__);
		
		err = cudaFree(r_cubics);
		handle_cuda_err(err, __LINE__);
		
		err = cudaFree(g_cubics);
		handle_cuda_err(err, __LINE__);
		
		err = cudaFree(b_cubics);
		handle_cuda_err(err, __LINE__);
		
		if (this_g != nullptr) {
			err = cudaFree(this_g);
			handle_cuda_err(err, __LINE__);
			
			this_g = nullptr;
		}
		
		#else
		delete[] positions;
		delete[] colors;
		
		delete[] r_cubics;
		delete[] g_cubics;
		delete[] b_cubics;
		#endif
	}
};

void save_img(int ind, uint8_t* img, int width, int height) {
	char buf[32];
	sprintf(buf, "out/%04d.ppm", ind);
	
	FILE* fout = fopen(buf, "wb");
	int hdr_len = sprintf(buf, "P6 %d %d 255 ", width, height);
	
	fwrite(buf, 1, hdr_len, fout);
	fwrite(img, 1, width*height*3, fout);
	
	fclose(fout);
}

__global__ void render_mandel(int num_iters, float sqr_escape_rad, int supersample_lvl, mat2 cam, palette* pal, uint8_t* img, int width, int height) {
	float dbail = 1E-6;
	
	#ifdef __CUDACC__
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	#else
	for (int x = 0; x < width; x++) {
		if (x % 20 == 0) printf("%.2f%%...\n", (float) x / width * 100);
		for (int y = 0; y < height; y++) {
	#endif
			//printf("%d, %d\n", x, y);
			int num_escaped = 0;
			int num_iters_to_escape = 0;
			float total_final_abs = 0;
			
			for (int a = 0; a < supersample_lvl; a++) {
				for (int b = 0; b < supersample_lvl; b++) {
					
					float C[2] = {(float) x + (float) a/supersample_lvl, (float) y + (float) b/supersample_lvl};
					cam.transform(C);
					
					float Z[2] = {C[0], C[1]};
					float Z_r_t = 0;
					
					float dZ[2] = {1, 0};
					float dZ_r_t;
					
					for (int i = 2; i <= num_iters; i++) {
						dZ_r_t = (dZ[0]*Z[0] - dZ[1]*Z[1]) * 2;
						dZ[1] = (dZ[0]*Z[1] + dZ[1]*Z[0]) * 2;
						dZ[0] = dZ_r_t;
						
						Z_r_t = Z[0]*Z[0] - Z[1]*Z[1] + C[0];
						Z[1] = Z[0]*Z[1]*2 + C[1];
						Z[0] = Z_r_t;
						
						float sqr_rad = Z[0]*Z[0] + Z[1]*Z[1];
						if (sqr_rad > sqr_escape_rad) {
							num_escaped++;
							total_final_abs += sqrt(sqr_rad);
							num_iters_to_escape += i;
							//printf("  %d, %d in %d\n", a, b, i);
							break;
						}
						
						float d_sqr_rad = dZ[0]*dZ[0] + dZ[1]*dZ[1];
						if (d_sqr_rad < dbail) {
							break;
						}
					}
				}
			}
			
			int super_lvl_sqr = supersample_lvl*supersample_lvl;
			float frac_escaped = (float) num_escaped / super_lvl_sqr;
			
			color ext_val;
			if (num_escaped > 0) {
				float avg_iters_to_escape = (float) num_iters_to_escape / num_escaped;
				float avg_final_abs = total_final_abs / num_escaped;
				float renorm_cnt = avg_iters_to_escape + 1 - log2(log2(avg_final_abs));
				
				ext_val = pal->evaluate(fmodf(pow(renorm_cnt, (float) 0.6) / 10, 1));
			}
			else {
				ext_val = color(255, 255, 255);
			}
			
			color int_val(0, 0, 0);
			
			//printf("  %d, %d, %d -> %.2f\n", num_iters_to_escape, num_escaped, num_iters, ext_val);
			
			color final_col = color::lerp(int_val, ext_val, frac_escaped);
			
			int ind = (y*width + x)*3;
			img[ind  ] = final_col.r;
			img[ind+1] = final_col.g;
			img[ind+2] = final_col.b;
			
	#ifndef __CUDACC__
		}
	}
	#endif
}

int main() {
	#ifdef __CUDACC__
	cudaError_t err;
	#endif
	
	int width = 320*8;
	int height = 120*8;
	
	int num_iters = 5000;
	float sqr_escape_rad = 8*8;
	
	// pow(supersample_lvl, 2) samples will be taken per pixel.
	// Supersampling does not increase memory usage.
	int supersample_lvl = 16;
	
	//mat2 cam(-1, 0, width, height, 0.24); // Entire Mandelbrot
		//mat2 cam(-1.14, 0.333, width, height, 2.46); // Branches
		//mat2 cam(-0.709, -0.352, width, height, 24.39); // Noise City 0.119531 x 0.616346, 0.315365 x 0.285577
			//mat2 cam(-0.724604, -0.349313, width, height, 8000); // Swirl 1 (good zoom, FPB @ Z=8000)
			//mat2 cam(-0.7165701, -0.3569452, width, height, 24.39); // Swirl 2 (good zoom, FPB @ Z=8000)
		//mat2 cam(-1.55, 0, width, height, 1); // Spire
		//mat2 cam(-0.523, 0.575, width, height, 0.48913); // Half
		
	// mat2 cam(-0.5755, 0.5747, width, height, 4000); // Half
	
	mat2 cam(0.19, -0.75, width, height, 1.695); // Half Spire
	
	// The zoom level that the animation approaches.
	// Has no impact while rendering frame 0.
	// Final frame renders at precisely this zoom level.
	float final_zoom = 0.45;
	
	// Number of frames in the animation.
	int num_frames = 1;
	
	// Portion of the animation to render.
	int start_frame = 0;
	int end_frame = num_frames;
	
	printf("Constructing Palette...\n");
	palette pal(5, 0, 0, 7, 100);
	
	// Default Palette
	// pal.add_color(0.16,    32, 107, 203);
	// pal.add_color(0.42,   237, 255, 255);
	// pal.add_color(0.6425, 255, 170,   0);
	// pal.add_color(0.8575,   0,   2,   0);
	
	// Brand Palette
	pal.add_color(0.16,    80, 60, 180);
	pal.add_color(0.42,   237, 255, 255);
	pal.add_color(0.6425, 255, 120,   180);
	pal.add_color(0.8575,   0,   2,   0);
	
	
	pal.cache_cubics();
	pal.debug("test.ppm");
	
	float start_zoom = cam.zoom;
	
	#ifdef __CUDACC__
		uint8_t* img; // = new uint8_t[width*height*3];
		err = cudaMallocManaged(&img, width*height*3);
		handle_cuda_err(err, __LINE__);
	#else
		uint8_t* img = new uint8_t[width*height*3];
	#endif
	
	float start = (float) clock() / CLOCKS_PER_SEC;
	
	for (int f = start_frame; f < end_frame; f++) {
		printf("Rendering frame %04d... ", f);
		float curr_zoom = start_zoom * pow((float) start_zoom / final_zoom, (float) -f / (num_frames - 1));
		//cam = mat2(-0.575305, 0.574783, width, height, curr_zoom);
		
		#ifdef __CUDACC__
			int block_edge = 8;
			dim3 block(block_edge, block_edge);
			dim3 grid(width/block_edge, height/block_edge);
			
			render_mandel<<<grid, block>>>(num_iters, sqr_escape_rad, supersample_lvl, cam, pal.get_gpu_ptr(), img, width, height);
			err = cudaGetLastError();
			handle_cuda_err(err, __LINE__);
			
			printf("Synchronizing... ");
			err = cudaDeviceSynchronize();
			handle_cuda_err(err, __LINE__);
		#else
			render_mandel(num_iters, sqr_escape_rad, supersample_lvl, cam, pal_p, img, width, height);
		#endif
		
		printf("Saving...\n");
		
		save_img(f, img, width, height);
	}
	
	float end = (float) clock() / CLOCKS_PER_SEC;
	printf("Complete in %.2f seconds.\n", end-start);
	
	// delete[] img;
	#ifdef __CUDACC__
		err = cudaFree(img);
		handle_cuda_err(err, __LINE__);
	#else
		delete[] img;
	#endif
	
	printf("Done.\n");
	return 0;
}