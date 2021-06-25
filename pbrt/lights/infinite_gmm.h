
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTS_INFINITE_GMM_H
#define PBRT_LIGHTS_INFINITE_GMM_H

// lights/infinite_gmm.h*
#include "pbrt.h"
#include "light.h"
#include "texture.h"
#include "shape.h"
#include "scene.h"
#include "mipmap.h"

namespace pbrt {

constexpr size_t GMM_TURB_DIM = 10;
constexpr size_t GMM_ELEV_DIM = 90;
constexpr size_t GMM_COMP_DIM = 5;

struct Gaussian2D {
	Float weight = 0.0f;
	Float mean_x = 0.0f;
	Float mean_y = 0.0f;
	Float sigma_x = 0.0f;
	Float sigma_y = 0.0f;
	Float volume_x = 0.0f;
	Float volume_y = 0.0f;

	Float Eval(Float x, Float y) const;
	Vector3f Sample(const Point2f &u) const;
};

class GMM2D {

public:

    
	Float cdf[GMM_COMP_DIM + 1] = { 0 };
	int comp_count          = GMM_COMP_DIM;
	Gaussian2D comps[GMM_COMP_DIM];

	Float Pdf(const Vector3f& v) const;
	Vector3f Sample(const Point2f &u) const;

	auto& operator[] (size_t index) {
		return comps[index];
	}
};

class SkyModel {
private:
    size_t active_elev_index = 0;
    size_t active_turb_index = 0;
    size_t elev_dim = 0;
    size_t turb_dim = 0;
    std::vector<std::vector<GMM2D>> gmms;
public:

	SkyModel(size_t turb_count, size_t elev_count) : 
		elev_dim(elev_count), turb_dim(turb_count) {
		gmms.resize(turb_count);
		for (auto& t : gmms) {
			t.resize(elev_count);
		}

	};
	SkyModel() : 
		SkyModel(GMM_TURB_DIM, GMM_ELEV_DIM) {
    };

    GMM2D& GetGMMByIndex(size_t turbidity_index, size_t elevation_index) {
        return gmms[turbidity_index][elevation_index];
    }

    void setActiveGMM(int turbidity, int elevation) {
        turbidity = Clamp(turbidity, 1, turb_dim);
        elevation = Clamp(elevation, 1, elevation);
        active_turb_index = size_t(turbidity) - 1u;
        active_elev_index = size_t(elevation) - 1u;
    }

    int getActiveTurbidity() {
        return active_turb_index + 1;
    }
    int getActiveElevation() {
        return active_elev_index + 1;
    }
    Gaussian2D& getGaussian(size_t turbidity, size_t elevation, size_t index) {
        turbidity = Clamp(turbidity, 1, turb_dim) - 1;
        elevation = Clamp(elevation, 1, elevation) - 1;
		return gmms[turbidity][elevation][index];
	}

	inline const GMM2D& GetActiveGMM() const {
		return gmms[active_turb_index][active_elev_index];
	}
};

// InfiniteAreaLight Declarations
class GMMInfiniteAreaLight : public Light {
  public:
    // InfiniteAreaLight Public Methods
	GMMInfiniteAreaLight(const Transform &LightToWorld, const Spectrum &power,
                         int nSamples, const std::string &texmap, const std::string &model,
						 int turbidity = 4, int elevation = 18);
    void Preprocess(const Scene &scene) {
        scene.WorldBound().BoundingSphere(&worldCenter, &worldRadius);
    }
    Spectrum Power() const;
    Spectrum Le(const RayDifferential &ray) const;
    Spectrum Sample_Li(const Interaction &ref, const Point2f &u, Vector3f *wi,
                       Float *pdf, VisibilityTester *vis) const;
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    Spectrum Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                       Ray *ray, Normal3f *nLight, Float *pdfPos,
                       Float *pdfDir) const;
    void Pdf_Le(const Ray &, const Normal3f &, Float *pdfPos,
                Float *pdfDir) const;

  private:
    // InfiniteAreaLight Private Data
    std::unique_ptr<MIPMap<RGBSpectrum>> Lmap;
    Point3f worldCenter;
    Float worldRadius;
	std::unique_ptr<Distribution2D> distribution;

	/* Will hold GMM_TURB_DIM * GMM_ELEV_DIM gmms */
	std::unique_ptr<SkyModel> sky_model;
};

std::shared_ptr<GMMInfiniteAreaLight> CreateGMMInfiniteLight(
    const Transform &light2world, const ParamSet &paramSet);

}  // namespace pbrt

#endif  // PBRT_LIGHTS_INFINITE_H
