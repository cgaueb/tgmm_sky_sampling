
/*
    Sampling Clear Sky Models using Truncated Gaussian Mixtures
    Authors: Vitsas Nick, Vardis Konstantinos, Papaioannou Georgios
    
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// lights/infinite_uniform.cpp*
#include "lights/infinite_gmm.h"
#include "imageio.h"
#include "paramset.h"
#include "sampling.h"
#include "stats.h"
#include "reflection.h"

#include <fstream>
#include <string>
#include <unordered_map>

namespace pbrt {

template<typename T>
static inline T Sign(T v) {
	return v < T(0) ? T(-1) : (v > T(0) ? T(1) : T(0));
}

static inline Float ApproxErf(float x) {
	Float sign_of = Sign(x);
	constexpr Float p = 0.47047f;
	constexpr Float a1 = 0.34802f;
	constexpr Float a2 = -0.09587f;
	constexpr Float a3 = 0.74785f;
	Float t = 1.0f / (1.0f + p * x * sign_of);
	Float tt = t * t;
	Float ttt = tt * t;
	Float xx = x * x;

	return sign_of * (1.0f - (a1 * t + a2 * tt + a3 * ttt) * expf(-xx));
}

static inline Float ApproxQuantile(Float x, Float mu = 0.0, Float sigma = 1.0) {
	/* Handle division by zero */
	if (x == 0.0f)
		return 0.0f;

	// Define break - points.
	PBRT_CONSTEXPR Float plow = 0.02425f;
	//static const float phigh = 1.0f - 0.02425f;

	// Coefficients in rational approximations
	PBRT_CONSTEXPR Float g_quantile_coeff_a[] = { -39.696830f, 220.946098f, -275.928510f, 138.357751f, -30.664798f, 2.506628f };

	PBRT_CONSTEXPR Float g_quantile_coeff_b[] = { -54.476098f, 161.585836f, -155.698979f, 66.801311f, -13.280681f };

	PBRT_CONSTEXPR Float g_quantile_coeff_c[] = { -0.007784894002f, -0.32239645f, -2.400758f, -2.549732f, 4.374664f, 2.938163f };

	PBRT_CONSTEXPR Float g_quantile_coeff_d[] = { 0.007784695709f, 0.32246712f, 2.445134f, 3.754408f };

	Float result = 0.0f;
	if (x < plow) {
		Float q = std::sqrt(-2.0f * std::log(x));

		result = (((((g_quantile_coeff_c[0] * q + g_quantile_coeff_c[1]) * q + g_quantile_coeff_c[2]) * q + g_quantile_coeff_c[3]) * q + g_quantile_coeff_c[4]) * q + g_quantile_coeff_c[5]) / ((((g_quantile_coeff_d[0] * q + g_quantile_coeff_d[1]) * q + g_quantile_coeff_d[2]) * q + g_quantile_coeff_d[3]) * q + 1.0f);
	}
	else {
		Float q = x - 0.5f;
		Float r = q * q;

		result = (((((g_quantile_coeff_a[0] * r + g_quantile_coeff_a[1]) * r + g_quantile_coeff_a[2]) * r + g_quantile_coeff_a[3]) * r + g_quantile_coeff_a[4]) * r + g_quantile_coeff_a[5]) * q / (((((g_quantile_coeff_b[0] * r + g_quantile_coeff_b[1]) * r + g_quantile_coeff_b[2]) * r + g_quantile_coeff_b[3]) * r + g_quantile_coeff_b[4]) * r + 1.0f);
	}
	return mu + sigma * result;
}

/* https://en.wikipedia.org/wiki/Normal_distribution */

/* PBRT offers its own implementation for ErfInv so we can choose */
static inline Float GaussQuantile(Float x, Float mu = 0.0f, Float sigma = 1.0f) {
	//return ApproxQuantile(x, mu, sigma)
	return mu + sigma * Sqrt2 * ErfInv(2 * x - 1);
}

/* PBRT offers its own implementation for Erf so we can choose */
static inline Float GaussCDF(Float x, Float mu = 0.0f, Float sigma = 1.0f) {
	//return 0.5f * (1.0f + ApproxErf((x - mu) / (sigma * Sqrt2)));
	return 0.5f * (1.0f + Erf((x - mu) / (sigma * Sqrt2)));
}

static inline Float GaussVolume(Float mu = 0.0f, Float sigma = 1.0f, Float low = -10.0f, Float high = 10.0f) {
	Float Fa = GaussCDF(low, mu, sigma);
	Float Fb = GaussCDF(high, mu, sigma);

	return (Fb - Fa);
}

static inline Float Gauss2DEval(Float x, Float y, Float mux = 0.0f, Float sigmax = 1.0f, Float muy = 0.0f, Float sigmay = 1.0f) {
	Float a = (x - mux) / sigmax;
	Float b = (y - muy) / sigmay;

	return (1.0f / (2 * Pi * sigmax * sigmay)) * expf(-0.5f * (a * a + b * b));
}

static inline Float
TruncatedGauss2DEval(Float x, Float y, Float mux = 0.0f, Float sigmax = 1.0f, float muy = 0.0f, float sigmay = 1.0f) {
	Float volume_x = GaussVolume(mux, sigmax, 0.0, 2 * Pi);
	Float volume_y = GaussVolume(muy, sigmay, 0.0, PiOver2);
	Float coverage = volume_x * volume_y;
	Float eval = Gauss2DEval(x, y, mux, sigmax, muy, sigmay);

	return (eval / coverage);
}

/* @see https://en.wikipedia.org/wiki/Inverse_transform_sampling#The_method */
static inline Float
GaussSample(Float u, Float mu = 0.0f, Float sigma = 1.0f, Float low = -10.0f, Float high = 10.0f) {
	Float Fa = GaussCDF(low, mu, sigma);
	Float Fb = GaussCDF(high, mu, sigma);
	Float v = Fa + (Fb - Fa) * u;

	return GaussQuantile(v, mu, sigma);
}

static std::unique_ptr<SkyModel> ReadModel(const std::string& model, int turbidity, int elevation) {
	auto sky_model = std::make_unique<SkyModel>();
	std::fstream fin;
	fin.open(model, std::ios::in);
	if (!fin.is_open()) return sky_model;

	std::string line, token;
	std::vector<std::string> col_names;
	
	if (!fin.eof()) {
		std::getline(fin, line);
		std::stringstream s(line);
		while (std::getline(s, token, ',')) {
			col_names.push_back(token);
		}
	}

	size_t elev_index = -1;
	size_t turb_index = -1;
	size_t gauss_index = 0;
	while (!fin.eof()) {
		// read an entire row and 
		// store it in a string variable 'line' 
		std::vector<std::string> tokens;
		std::getline(fin, line);
		std::stringstream s(line);
		while (std::getline(s, token, ',')) {
			tokens.push_back(token);
		}
		if (tokens.size() == 0) continue;

		std::unordered_map<std::string, Float> params;
		for (size_t index = 0; index < tokens.size(); index++) {
			params[col_names[index]] = std::atof(tokens[index].c_str());
		}

		size_t prev_turb_index = turb_index;
		turb_index = size_t(params["Turbidity"]);
		size_t prev_elev_index = elev_index;
		elev_index = size_t(params["Elevation"]);
		if (turb_index != prev_turb_index || elev_index != prev_elev_index) {
			gauss_index = 0;
		}

		Gaussian2D& gaussian = sky_model->getGaussian(turb_index, elev_index, gauss_index);
		gaussian.weight = params["Weight"];
		gaussian.mean_x = params["Mean X"];
		gaussian.mean_y = params["Mean Y"];
		gaussian.sigma_x = params["Sigma X"];
		gaussian.sigma_y = params["Sigma Y"];

		/* Cache volumes */
		gaussian.volume_x = GaussVolume(gaussian.mean_x, gaussian.sigma_x, 0.0, 2 * Pi);
		gaussian.volume_y = GaussVolume(gaussian.mean_y, gaussian.sigma_y, 0.0, PiOver2);

		gauss_index++;
	}

	for (size_t turb_index = 0; turb_index < GMM_TURB_DIM; turb_index++) {
		for (size_t elev_index = 0; elev_index < GMM_ELEV_DIM; elev_index++) {
			GMM2D& gmm = sky_model->GetGMMByIndex(turb_index, elev_index);

			/* Cache CDF */
			for (int i = 1; i < GMM_COMP_DIM + 1; ++i)
				gmm.cdf[i] = gmm.cdf[i - 1] + gmm.comps[i - 1].weight;
		}
	}

	sky_model->setActiveGMM(turbidity, elevation);

	const GMM2D& gmm = sky_model->GetActiveGMM();
    printf("GMM Sky Model. Active Turbidity: %d, Elevation: %d\n", turbidity, elevation);
	printf("Gaussian 2D [Weight, MeanX, SigmaX, MeanY, SigmaY]\n");
	for (int i = 0; i < gmm.comp_count; ++i) {
		const Gaussian2D& gauss = gmm.comps[i];
		printf("Gaussian 2D [%f %f %f %f %f]\n", gauss.weight, gauss.mean_x, gauss.sigma_x, gauss.mean_y, gauss.sigma_y);
	}

	fin.close();

	return sky_model;
}

// InfiniteAreaLight Method Definitions
GMMInfiniteAreaLight::GMMInfiniteAreaLight(const Transform &LightToWorld,
                                     const Spectrum &L, int nSamples,
                                     const std::string &texmap,
									 const std::string &model,
									 int turbidity, int elevation)
    : Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(),
            nSamples) {
    // Read texel data from _texmap_ and initialize _Lmap_
    Point2i resolution;
    std::unique_ptr<RGBSpectrum[]> texels(nullptr);
    if (texmap != "") {
        texels = ReadImage(texmap, &resolution);
        if (texels)
            for (int i = 0; i < resolution.x * resolution.y; ++i)
                texels[i] *= L.ToRGBSpectrum();
    }
	if (model != "") {
		sky_model = ReadModel(model, turbidity, elevation);
	}
    if (!texels) {
        resolution.x = resolution.y = 1;
        texels = std::unique_ptr<RGBSpectrum[]>(new RGBSpectrum[1]);
        texels[0] = L.ToRGBSpectrum();
    }
    Lmap.reset(new MIPMap<RGBSpectrum>(resolution, texels.get()));

    // Initialize sampling PDFs for infinite area light

    // Compute scalar-valued image _img_ from environment map
    int width = 2 * Lmap->Width(), height = 2 * Lmap->Height();
    std::unique_ptr<Float[]> img(new Float[width * height]);
    float fwidth = 0.5f / std::min(width, height);
    ParallelFor(
        [&](int64_t v) {
            Float vp = (v + .5f) / (Float)height;
            Float sinTheta = std::sin(Pi * (v + .5f) / height);
            for (int u = 0; u < width; ++u) {
                Float up = (u + .5f) / (Float)width;
                img[u + v * width] = Lmap->Lookup(Point2f(up, vp), fwidth).y();
                img[u + v * width] *= sinTheta;
            }
        },
        height, 32);

    // Compute sampling distributions for rows and columns of image
    distribution.reset(new Distribution2D(img.get(), width, height));
}

Spectrum GMMInfiniteAreaLight::Power() const {
    return Pi * worldRadius * worldRadius *
           Spectrum(Lmap->Lookup(Point2f(.5f, .5f), .5f),
                    SpectrumType::Illuminant);
}

Spectrum GMMInfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Spectrum(Lmap->Lookup(st), SpectrumType::Illuminant);
}

Spectrum GMMInfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                      Vector3f *wi, Float *pdf,
                                      VisibilityTester *vis) const {
	const auto& gmm = sky_model->GetActiveGMM();

	*wi = gmm.Sample(u);
	*pdf = gmm.Pdf(*wi);
	if (*pdf == 0) return Spectrum(0.f);

	Float theta = SphericalTheta(*wi), phi = SphericalPhi(*wi);
	Point2f uv(phi * Inv2Pi, theta * InvPi);

	*wi = LightToWorld(*wi);

	// Return radiance value for infinite light direction
	*vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
		ref.time, mediumInterface));

	return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

Float GMMInfiniteAreaLight::Pdf_Li(const Interaction &, const Vector3f &w) const {
	return sky_model->GetActiveGMM().Pdf(WorldToLight(w));
}

Spectrum GMMInfiniteAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                      Float time, Ray *ray, Normal3f *nLight,
                                      Float *pdfPos, Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    // Compute direction for infinite light sample ray
    Point2f u = u1;

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution->SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);
    Float theta = uv[1] * Pi, phi = uv[0] * 2.f * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    Vector3f d =
        -LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    *nLight = (Normal3f)d;

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * -d, d, Infinity, time);

    // Compute _InfiniteAreaLight_ ray PDFs
    *pdfDir = sinTheta == 0 ? 0 : mapPdf / (2 * Pi * Pi * sinTheta);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

void GMMInfiniteAreaLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                               Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    Vector3f d = -WorldToLight(ray.d);
    Float theta = SphericalTheta(d), phi = SphericalPhi(d);
    Point2f uv(phi * Inv2Pi, theta * InvPi);
    Float mapPdf = distribution->Pdf(uv);
    *pdfDir = mapPdf / (2 * Pi * Pi * std::sin(theta));
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

Float Gaussian2D::Eval(Float x, Float y) const {
	Float eval = Gauss2DEval(x, y, mean_x, sigma_x, mean_y, sigma_y);
	return (eval / (volume_x * volume_y));
}

Vector3f Gaussian2D::Sample(const Point2f &u) const {
	return {};
}

Float GMM2D::Pdf(const Vector3f& v) const {
	if (CosTheta(v) < 0) return 0.0f;

	Float theta = PiOver2 - SphericalTheta(v);
	Float phi = SphericalPhi(v);
	Float pdf = 0.0f;
	for (size_t comp = 0; comp < GMM_COMP_DIM; comp++) {
		const Gaussian2D& gauss = comps[comp];

		Float component_pdf = 0.0f;
		//component_pdf += TruncatedGauss2DEval(phi, theta, gauss.mean_x, gauss.sigma_x, gauss.mean_y, gauss.sigma_y);
		component_pdf += gauss.Eval(phi, theta);
		component_pdf *= gauss.weight;
		pdf += component_pdf;
	}

	Float sintheta = std::sin(SphericalTheta(v));
	if (sintheta == 0) return 0.0;

	return pdf / sintheta;
}

Vector3f GMM2D::Sample(const Point2f &u) const {
	int comp = 0;
	for (int i = 1; i < comp_count + 1; ++i) {
		if (u.x < cdf[i]) {
			comp = i - 1;
			break;
		}
	}

	/*
	 * Rescale from [ cdf[comp + 0], cdf[comp + 1] ] to [ 0, 1 ]
	 * @see http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions.html#SamplingBSDFs
  	 * @see https://math.stackexchange.com/a/914843
	 */
	Point2f u_remapped = { std::min((u.x - cdf[comp + 0]) * (1.0f / (cdf[comp + 1] - cdf[comp + 0])), OneMinusEpsilon), u.y };
	
	const auto& gauss = comps[comp];
	Float phi = GaussSample(u_remapped.x, gauss.mean_x, gauss.sigma_x, 0, 2 * Pi);
	Float theta = PiOver2 - GaussSample(u_remapped.y, gauss.mean_y, gauss.sigma_y, 0, PiOver2);

	Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
	Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);

	return Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

std::shared_ptr<GMMInfiniteAreaLight> CreateGMMInfiniteLight(
	const Transform &light2world, const ParamSet &paramSet) {
	Spectrum L = paramSet.FindOneSpectrum("L", Spectrum(1.0));
	Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
	std::string texmap = paramSet.FindOneFilename("mapname", "");
	std::string model = paramSet.FindOneFilename("model", "");
	int nSamples = paramSet.FindOneInt("samples",
		paramSet.FindOneInt("nsamples", 1));
	int turbidity = paramSet.FindOneInt("turbidity",
		paramSet.FindOneInt("turbidity", 4));
	int elevation = paramSet.FindOneInt("elevation",
		paramSet.FindOneInt("elevation", 18));
	if (PbrtOptions.quickRender) nSamples = std::max(1, nSamples / 4);
	return std::make_shared<GMMInfiniteAreaLight>(light2world, L * sc, nSamples,
		texmap, model, turbidity, elevation);
}

}  // namespace pbrt
