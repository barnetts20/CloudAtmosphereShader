struct CloudFunctions {
    #define MAX_ITER 600
    //Shared/Derived parameters
    float2 SCREEN_UV;
    float3 CAMERA_POSITION;
    float3 CAMERA_VECTOR;
    float3 PLANET_POSITION;
    float3 LIGHT_POSITION;
    float3 LIGHT_DIRECTION;
    float LIGHT_INTENSITY;
    float PLANET_RADIUS;
    float SCENE_DEPTH;
    float3 WORLD_POSITION;
    float3 PIXEL_NORMAL;
    float3 SCENE_COLOR;
    //Atmosphere Parameters
    float ATMO_THICKNESS;
    float ATMO_RADIUS;
    float ATMO_OFFSET;
    float3 ATMO_RAY_BETA;
    float3 ATMO_MIE_BETA;
    float3 ATMO_ABSORPTION_BETA;
    float3 ATMO_AMBIENT_BETA;
    float ATMO_G;
    float ATMO_HEIGHT_RAY;
    float ATMO_HEIGHT_MIE;
    float ATMO_HEIGHT_ABSORPTION;
    float ATMO_ABSORPTION_FALLOFF;
    int ATMO_DENSITY_STEPS;
    int ATMO_LIGHT_STEPS;   
    float3 ATMO_SKYLIGHT_MULTIPLIER;
    //Cloud parameters
    float3 CLOUD_COLOR;
    float CLOUD_HEIGHT;
    float CLOUD_THICKNESS;
    float CLOUD_MIN_RADIUS;
    float CLOUD_MAX_RADIUS;
    float CLOUD_ABSORPTION_TO_LIGHT;
    float CLOUD_ABSORPTION_TO_CLOUD;
    float CLOUD_DARKNESS_THRESHOLD;
    float CLOUD_G;
    float CLOUD_AMBIENT;
    float CLOUD_SHADOW_SOFTNESS;
    float CLOUD_SCATTER_FADE;

    float CLOUD_HEIGHT_FALLOFF;
    float CLOUD_DENSITY_FALLOFF;
    float CLOUD_DENSITY_FLOOR;
    float CLOUD_NOISE_SCALE;
    float3 CLOUD_NOISE_OFFSET;
    float4 CLOUD_DISTORTION_SCALE_WEIGHT;
    float4 CLOUD_SHAPE_SCALE_WEIGHT;
    float4 CLOUD_DETAIL_SCALE_WEIGHT;
    float4 CLOUD_MICRO_SCALE_WEIGHT;

    float CLOUD_STEP_SCALE;    
    int CLOUD_DENSITY_STEPS;
    int CLOUD_LIGHT_STEPS;

    Texture3D CLOUD_VOLUME_TEXTURE;
    SamplerState CLOUD_VOLUME_TEXTURE_SAMPLER;
    Texture2D CLOUD_BLUE_NOISE_TEXTURE;
    SamplerState CLOUD_BLUE_NOISE_TEXTURE_SAMPLER;

    //Ray Sphere Intersect, X = distance to first intersect, Y = distance to second intersect
    float2 SphereIntersections(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereRadius) {
        float3 m = rayOrigin - sphereCenter;
        float b = dot(m, rayDir);
        float c = dot(m, m) - sphereRadius * sphereRadius;
        float discriminant = b * b - c;        
        if (discriminant < 0) {
            return float2(-1.0, -1.0); // No intersection
        }
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = -b - sqrtDiscriminant;
        float t2 = -b + sqrtDiscriminant;
        return float2(t1, t2);
    }

    //Ray sphere used in the atmosphere calculations, returns all values not just ray distances
    float4 SphereIntersections2(float3 ray_direction, float3 start_position, float radius){
        float a = dot(ray_direction, ray_direction);
        float b = 2.0 * dot(ray_direction, start_position);
        float c = dot(start_position, start_position) - (radius * radius);
        float d = (b * b) - 4.0 * a * c;        
        return float4(a,b,c,d);
    }

    //Calculates the ray segments needed to trace the current path through the shell formed by the min and max cloud radi
    //Returns a float4 with X = distance to ray1 start, Y = distance to ray1 end, Z = distance to ray2 start, A = distance to ray2 end
    float4 GetCloudTraceSegments() {
        float4 result = float4(-1.0, -1.0, -1.0, -1.0);
        if(length(CAMERA_POSITION - PLANET_POSITION) < PLANET_RADIUS) return result;
        float2 outerIntersects = SphereIntersections(CAMERA_POSITION, CAMERA_VECTOR, PLANET_POSITION, CLOUD_MAX_RADIUS);
        float2 innerIntersects = SphereIntersections(CAMERA_POSITION, CAMERA_VECTOR, PLANET_POSITION, CLOUD_MIN_RADIUS);
        float enterOuter = outerIntersects.x;
        float exitOuter = min(outerIntersects.y, SCENE_DEPTH);
        float enterInner = innerIntersects.x;
        float exitInner = min(innerIntersects.y, SCENE_DEPTH);

        if (enterOuter >= 0) {
            if (enterInner >= 0) {
                result.x = enterOuter;
                result.y = exitInner;
                if (exitInner < exitOuter) {
                    result.z = enterInner;
                    result.a = exitOuter;
                }
            } else {
                result.x = enterOuter;
                result.y = exitOuter;
            }
        } else if (enterInner >= 0) {
            result.x = enterInner;
            result.y = exitOuter;
        } else {
            if (exitOuter > 0) {
                result.x = 0;
                result.y = exitOuter;
            }
        }
        if (enterOuter < 0 && exitOuter > 0) result.x = 0;
        if (enterInner < 0 && exitInner > 0 && enterOuter < 0) result.z = 0;
        return result;
    }

    //Calculates a fade factor to fade out the cloud density near the top and bottom of the cloud "shell"
    float HeightCoefficient(float3 position){
        float radius = length(position - PLANET_POSITION);
        float shell_thickness = (CLOUD_MAX_RADIUS - CLOUD_MIN_RADIUS);
        float ramp_up_end = CLOUD_MIN_RADIUS + shell_thickness * .2;
        float ramp_down_start = CLOUD_MAX_RADIUS - shell_thickness * .6;
        float fade_multiplier = 1.0; 
        if(radius < CLOUD_MIN_RADIUS || radius > CLOUD_MAX_RADIUS){
            return 0.0;
        }
        if (radius < ramp_up_end) {
            fade_multiplier = (radius - CLOUD_MIN_RADIUS) / (ramp_up_end - CLOUD_MIN_RADIUS);
        } else if (radius > ramp_down_start) {
            fade_multiplier = (CLOUD_MAX_RADIUS - radius) / (CLOUD_MAX_RADIUS - ramp_down_start);
        }
        return pow(clamp(fade_multiplier,0.0,1.0), CLOUD_HEIGHT_FALLOFF);
    }
    
    //Calculates the darknening from the shadow cone of the planet
    float ShadowFactor(float3 position) {
        float3 toLight = normalize(LIGHT_POSITION - position);
        float2 planetIntersect = SphereIntersections(position, toLight, PLANET_POSITION, PLANET_RADIUS);

        // Only consider shadows if there is an intersection behind the sample position
        if (planetIntersect.x <= 0 && planetIntersect.y <= 0) return 1.0; 

        // Calculate the directional vector from the position to the planet center and its distance
        float3 toPlanetCenter = normalize(PLANET_POSITION - position);
        float distanceToPlanetCenter = length(PLANET_POSITION - position);

        // Calculate the cosine of the critical angle for the shadow transition
        float angleCosineThreshold = sqrt(1 - (PLANET_RADIUS * PLANET_RADIUS) / (distanceToPlanetCenter * distanceToPlanetCenter));

        // Dot product to determine the angle between the light direction and the vector to the planet's center
        float dotProduct = dot(toLight, toPlanetCenter);

        // Adjust shadowTransition to center the softening around the shadow cone boundary
        float shadowSoftnessParameter = CLOUD_SHADOW_SOFTNESS / 2.0; // Adjust this parameter based on desired softness
        float lowerLimit = angleCosineThreshold - shadowSoftnessParameter;
        float upperLimit = angleCosineThreshold + shadowSoftnessParameter;
        float shadowTransition = smoothstep(lowerLimit, upperLimit, dotProduct);

        // Interpolate between fully lit (1.0) and fully shadowed (0) based on the transition value
        return lerp(1.0, 0, shadowTransition);
    }

    //Phase Function
    float HenyeyGreenstein(float3 lightDir, float3 viewDir, float g) {
        float cosTheta = dot(lightDir, viewDir);
        float gSquared = g * g;
        return (1.0 - gSquared) / (4.0 * 3.14159265 * pow(1.0 + gSquared - 2.0 * g * cosTheta, 1.5));
    }

    //Samples the cloud volume noise density formula at a given point
    float SampleCloudDensity(float3 position){
        float mod_noise_scale = CLOUD_MAX_RADIUS * CLOUD_NOISE_SCALE;
        float3 dst_scale = CLOUD_DISTORTION_SCALE_WEIGHT.xyz * mod_noise_scale;
        float3 shp_scale = CLOUD_SHAPE_SCALE_WEIGHT.xyz * mod_noise_scale;
        float3 dtl_scale = CLOUD_DETAIL_SCALE_WEIGHT.xyz * mod_noise_scale;
        float3 mcr_scale = CLOUD_MICRO_SCALE_WEIGHT.xyz * mod_noise_scale;
        float4 speed_coef = float4(.5, 1, 1.1, -.5);
        float4 distortion_samp = (Texture3DSample(CLOUD_VOLUME_TEXTURE, CLOUD_VOLUME_TEXTURE_SAMPLER, position/dst_scale + CLOUD_NOISE_OFFSET * speed_coef.x) - 0.5) * CLOUD_DISTORTION_SCALE_WEIGHT.a;
        float4 shape_samp = Texture3DSample(CLOUD_VOLUME_TEXTURE, CLOUD_VOLUME_TEXTURE_SAMPLER, position/shp_scale + CLOUD_NOISE_OFFSET  * speed_coef.y + distortion_samp.xyz) * CLOUD_SHAPE_SCALE_WEIGHT.a;
        float4 detail_samp = Texture3DSample(CLOUD_VOLUME_TEXTURE, CLOUD_VOLUME_TEXTURE_SAMPLER, position/dtl_scale + CLOUD_NOISE_OFFSET * speed_coef.z) * CLOUD_DETAIL_SCALE_WEIGHT.a;
        float4 micro_samp = Texture3DSample(CLOUD_VOLUME_TEXTURE, CLOUD_VOLUME_TEXTURE_SAMPLER, (position/mcr_scale + CLOUD_NOISE_OFFSET * speed_coef.a) * CLOUD_MICRO_SCALE_WEIGHT.a);  
        float4 final_samp = (shape_samp + detail_samp + micro_samp) / (CLOUD_SHAPE_SCALE_WEIGHT.a + CLOUD_DETAIL_SCALE_WEIGHT.a + CLOUD_MICRO_SCALE_WEIGHT.a);

        float d = (final_samp.x * 3.0 + final_samp.y * 2.0 + final_samp.z * 1.0) / 6.0 * HeightCoefficient(position);
        d -= CLOUD_DENSITY_FLOOR;
        d = max(d, 0.0);
        return pow(d, CLOUD_DENSITY_FALLOFF);
    }

    //Ray march function to do our cloud light marching
    float LightMarch(float3 position, float densityStepSize){
        float stepSize = float(CLOUD_DENSITY_STEPS) / float(CLOUD_LIGHT_STEPS) * densityStepSize;
        float3 lightDirection = normalize(LIGHT_POSITION - position);
        float totalDensity = 0;
        for(int i = 0; i < CLOUD_LIGHT_STEPS; i++){
            position += lightDirection * stepSize;
            float density = SampleCloudDensity(position) * stepSize;
            //Accumulate light density
            totalDensity += max(0, density);  
        }
        //Beer's law
        float transmittance = exp(-totalDensity * CLOUD_ABSORPTION_TO_LIGHT);
        return CLOUD_DARKNESS_THRESHOLD + transmittance * (1 - CLOUD_DARKNESS_THRESHOLD);
    }

    //Trace function to trace a single cloud ray
    float4 CloudMarch(float3 startPosition, float stepSize, float distanceLimit, float4 cloudAccumulation){
        float distanceTravelled = 0;
        int iter = 0;
        
        while(distanceTravelled < distanceLimit){
            float3 samplePosition = startPosition + CAMERA_VECTOR * distanceTravelled;
            float density = SampleCloudDensity(samplePosition);
            if(density > 0){
                float lightTransmittance = LightMarch(samplePosition, stepSize);
                float shadowFactor = ShadowFactor(samplePosition);
                //Each step with density accumulates: cloud density * step size * transmittance * light transmittance * phase * shadow factor
                cloudAccumulation.xyz += density * stepSize * cloudAccumulation.a * lightTransmittance * HenyeyGreenstein(LIGHT_DIRECTION, CAMERA_VECTOR, CLOUD_G * shadowFactor) * max(shadowFactor, CLOUD_AMBIENT) ;
                //And adjusts transmittance
                cloudAccumulation.a *= exp(-density * stepSize * CLOUD_ABSORPTION_TO_CLOUD);
                if(cloudAccumulation.a < .01) break;
            }
            distanceTravelled += stepSize;
            iter++;
            if(iter > MAX_ITER) break; //Hard cap on iterations
        }
        return cloudAccumulation;
    }

    //Traces any needed cloud rays for a given GetCloudTraceSegments output
    float4 AccumulateClouds(){
        float4 segments = GetCloudTraceSegments();
        //step size + blue noise to hide slices
        float stepSize = ((CLOUD_MAX_RADIUS - CLOUD_MIN_RADIUS) * CLOUD_STEP_SCALE) / float(CLOUD_DENSITY_STEPS);
        float blueSamp = Texture2DSample(CLOUD_BLUE_NOISE_TEXTURE, CLOUD_BLUE_NOISE_TEXTURE_SAMPLER, SCREEN_UV * 10);
        float blueFactor = 1.0;
        float blueOffset = (blueSamp - .5) * blueFactor * stepSize;
        stepSize += blueOffset;

        float4 cloudAccumulation = float4(0.0, 0.0, 0.0, 1.0); //xyz = light accumulation, a = transmittance
        //Trace applicable segments
        float segLength1 = length(segments.y - segments.x);
        float segLength2 = length(segments.a - segments.z);
        if(segLength1 > 0) cloudAccumulation = CloudMarch(CAMERA_POSITION + CAMERA_VECTOR * segments.x, stepSize, segLength1, cloudAccumulation);
        if(segLength2 > 0) cloudAccumulation = CloudMarch(CAMERA_POSITION + CAMERA_VECTOR * segments.z, stepSize, segLength2, cloudAccumulation);  
        cloudAccumulation.xyz = cloudAccumulation.xyz * CLOUD_COLOR;
        return cloudAccumulation;
    }

    struct atmoOut{
        float3 color;
        float3 opacity;
    };

    //Calculate atmosphere and cloud scattering
    atmoOut CalculateScattering(float3 start_position, float3 ray_direction, float max_distance, float3 scene_color, int d_steps, int l_steps) {
        atmoOut result;
        result.color = scene_color;
        start_position -= PLANET_POSITION; // Shift the position so that we can place our atmosphere at arbitrary locations instead of 0,0,0
        float offsetPlanetRadius = PLANET_RADIUS - (PLANET_RADIUS * ATMO_OFFSET);
        //Eventually it would be nice to come back and replace these inline ray-sphere intersects with our function. not doing it yet cause im scared ill break it
        float4 rs1 = SphereIntersections2(ray_direction, start_position, ATMO_RADIUS);
        if (rs1.a < 0.0) return result;//float4(result,0.0);        
        float2 ray_length = float2(
            max((-rs1.y - sqrt(rs1.a)) / (2.0 * rs1.x), 0.0),
            min((-rs1.y + sqrt(rs1.a)) / (2.0 * rs1.x), max_distance)
        );
        if (ray_length.x > ray_length.y) return result;//float4(result,0.0);
        // prevent the mie glow from appearing if there's an object in front of the camera
        bool allow_mie = max_distance > ray_length.y;
        // make sure the ray is no longer than allowed
        ray_length.y = min(ray_length.y, max_distance);
        ray_length.x = max(ray_length.x, 0.0);
        // get the step size of the ray
        float step_size_i = (ray_length.y - ray_length.x) / float(d_steps);
        
        // next, set how far we are along the ray, so we can calculate the position of the sample
        // if the camera is outside the atmosphere, the ray should start at the edge of the atmosphere
        // if it's inside, it should start at the position of the camera
        // the min statement makes sure of that
        float ray_pos_i = ray_length.x + step_size_i * 0.5;
        
        // these are the values we use to gather all the scattered light
        float3 total_ray = float3(0.0,0.0,0.0); // for rayleigh
        float3 total_mie = float3(0.0,0.0,0.0); // for mie
        
        // initialize the optical depth. This is used to calculate how much air was in the ray
        float3 opt_i = float3(0.0,0.0,0.0);
        
        // also init the scale height, avoids some float2's later on
        float2 scale_height = float2(ATMO_HEIGHT_RAY, ATMO_HEIGHT_MIE);
        
        // Calculate the Rayleigh and Mie phases.
        // This is the color that will be scattered for this ray
        // mu, mumu and gg are used quite a lot in the calculation, so to speed it up, precalculate them
        float mu = dot(ray_direction, LIGHT_DIRECTION);
        float mumu = mu * mu;
        float gg = ATMO_G * ATMO_G;
        float phase_ray = 3.0 / (50.2654824574 /* (16 * pi) */) * (1.0 + mumu);
        float phase_mie = allow_mie ? 3.0 / (25.1327412287 /* (8 * pi) */) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * ATMO_G, 1.5) * (2.0 + gg)) : 0.0;

        // now we need to sample the 'primary' ray. this ray gathers the light that gets scattered onto it
        for (int i = 0; i < d_steps; ++i) {
            // calculate where we are along this ray
            float3 pos_i = start_position + ray_direction * ray_pos_i;
            float3 pos_e = start_position + ray_direction * (ray_pos_i + step_size_i);
            // and how high we are above the surface
            float height_i = length(pos_i) - offsetPlanetRadius;            
            // now calculate the density of the particles (both for rayleigh and mie)
            float3 density = float3(exp(-height_i / scale_height), 0.0);            
            // and the absorption density. this is for ozone, which scales together with the rayleigh, 
            // but absorbs the most at a specific height, so use the sech function for a nice curve falloff for this height
            // clamp it to avoid it going out of bounds. This prevents weird black spheres on the night side
            float denom = (ATMO_HEIGHT_ABSORPTION - height_i) / ATMO_ABSORPTION_FALLOFF;
            density.z = (1.0 / (denom * denom + 1.0)) * density.x;            
            // multiply it by the step size here
            density *= step_size_i;
            // Add these densities to the optical depth, so that we know how many particles are on this ray.
            opt_i += density;
            // Calculate the step size of the light ray.
            // again with a ray sphere intersect
            // a, b, c and d are already defined
            float4 rs2 = SphereIntersections2(LIGHT_DIRECTION, pos_i, ATMO_RADIUS);
            float step_size_l = (-rs2.y + sqrt(rs2.a)) / (2.0 * rs2.x * float(l_steps));
            float ray_pos_l = step_size_l * 0.5;
            float3 opt_l = float3(0.0,0.0,0.0);                
            // now sample the light ray
            // this is similar to what we did before
            for (int l = 0; l < ATMO_LIGHT_STEPS; ++l) {
                // calculate where we are along this ray
                float3 pos_l = pos_i + LIGHT_DIRECTION * ray_pos_l;
                float height_l = length(pos_l) - offsetPlanetRadius;
                // first, set the density for ray and mie
                float3 density_l = float3(exp(-height_l / scale_height), 0.0);
                // then, the absorption
                float denom = (ATMO_HEIGHT_ABSORPTION - height_l) / ATMO_ABSORPTION_FALLOFF;
                density_l.z = (1.0 / (denom * denom + 1.0)) * density_l.x;
                // multiply the density by the step size
                density_l *= step_size_l;
                // and add it to the total optical depth
                opt_l += density_l;
                // and increment where we are along the light ray.
                ray_pos_l += step_size_l;                
            }
            // Now we need to calculate the attenuation
            float3 attn = exp(-ATMO_RAY_BETA * (opt_i.x + opt_l.x) - ATMO_MIE_BETA * (opt_i.y + opt_l.y) - ATMO_ABSORPTION_BETA * (opt_i.z + opt_l.z));
            // accumulate the scattered light (how much will be scattered towards the camera)
            total_ray += density.x * attn;
            total_mie += density.y * attn;
            // and increment the position on this ray
            ray_pos_i += step_size_i;            
        }
        // calculate how much light can pass through the atmosphere
        float3 opacity = exp(-(ATMO_MIE_BETA * opt_i.y + ATMO_RAY_BETA * opt_i.x + ATMO_ABSORPTION_BETA * opt_i.z));
        float3 atmosphere_light = (
                phase_ray * ATMO_RAY_BETA * total_ray // rayleigh color
                + phase_mie * ATMO_MIE_BETA * total_mie // mie
                + opt_i.x * ATMO_AMBIENT_BETA // and ambient
        );
        float3 mod_atmosphere_light = atmosphere_light;
        result.color = mod_atmosphere_light.xyz;
        result.opacity = opacity;
        return result;
    }

    //Provides a slight light influence on the ground shading
    float3 Skylight() {
        // slightly bend the surface normal towards the light direction
        float3 skylight_color = float3(0.0,0.0,0.0);
        float3 surface_normal = normalize(lerp(PIXEL_NORMAL, LIGHT_DIRECTION, .5));
        if(length(WORLD_POSITION - CAMERA_POSITION) > length(PLANET_POSITION - CAMERA_POSITION)){
            return float3(0.0,0.0,0.0);
        }
        atmoOut skylight_scatter = CalculateScattering(
            WORLD_POSITION,
            surface_normal,
            ATMO_RADIUS * 3,
            SCENE_COLOR,
            ATMO_LIGHT_STEPS,
            ATMO_LIGHT_STEPS
        );
        skylight_color = skylight_scatter.color + SCENE_COLOR.xyz * skylight_scatter.opacity * ATMO_SKYLIGHT_MULTIPLIER;
        return skylight_color;
    }
    
    // Calculates a ramp value based on camera's altitude relative to the cloud layer, used to blend the atmosphere scattering with clouds when near the surface
    float CameraAltitudeRamp(float rampEndHeight) {
        float altitudeAbovePlanet = length(CAMERA_POSITION - PLANET_POSITION) - PLANET_RADIUS;
        float rampStart = CLOUD_MIN_RADIUS - PLANET_RADIUS;
        float rampEnd = CLOUD_MIN_RADIUS - rampEndHeight - PLANET_RADIUS;
        float normalizedAltitude = saturate((altitudeAbovePlanet - rampEnd) / (rampStart - rampEnd));
        return saturate(lerp(1.0, 0.0, normalizedAltitude));
    }

    //Composite our scene color and atmosphere and cloud outputs
    float3 MainImage() {
        float3 skylight_color = Skylight();        
        atmoOut atmosphereScattering = CalculateScattering(CAMERA_POSITION, CAMERA_VECTOR, SCENE_DEPTH, skylight_color, ATMO_DENSITY_STEPS, ATMO_LIGHT_STEPS);
        float4 cloudAccumulation = AccumulateClouds();
        float3 cloudColor = cloudAccumulation.xyz;
        float3 cloudAtmoColor = float3(atmosphereScattering.color + cloudColor * atmosphereScattering.opacity);     
        cloudColor = clamp(lerp(cloudColor, cloudAtmoColor, CameraAltitudeRamp((CLOUD_MIN_RADIUS - PLANET_RADIUS) * CLOUD_SCATTER_FADE)),0,1);
        float3 cloudSceneColor = SCENE_COLOR * cloudAccumulation.a + cloudColor;
        float3 atmoSceneColor = (atmosphereScattering.color * LIGHT_INTENSITY + skylight_color.xyz * atmosphereScattering.opacity);
        float3 finalSceneColor = 1.0 - exp(-atmoSceneColor);
        finalSceneColor = finalSceneColor * cloudAccumulation.a + cloudSceneColor;
        return finalSceneColor;
    }    
};

//Initialize struct
CloudFunctions cf;
//External/Derived parameters
cf.SCREEN_UV = screenUV;
cf.CAMERA_POSITION = cameraPosition;
cf.CAMERA_VECTOR = -cameraVector;
cf.PLANET_POSITION = planetPosition;
cf.LIGHT_POSITION = lightPosition;
cf.LIGHT_DIRECTION = normalize(cf.LIGHT_POSITION - cf.PLANET_POSITION);
cf.LIGHT_INTENSITY = lightIntensity;
cf.PLANET_RADIUS = planetRadius;
cf.SCENE_DEPTH = sceneDepth;
cf.WORLD_POSITION = worldPosition;
cf.PIXEL_NORMAL = pixelNormal;
cf.SCENE_COLOR = sceneColor;
//Atmosphere parameters
cf.ATMO_THICKNESS = atmoThickness;
cf.ATMO_RADIUS = cf.PLANET_RADIUS + cf.ATMO_THICKNESS;
cf.ATMO_OFFSET = atmoOffset;
cf.ATMO_RAY_BETA = atmoRayBeta;
cf.ATMO_MIE_BETA = atmoMieBeta;
cf.ATMO_ABSORPTION_BETA = atmoAbsorptionBeta;
cf.ATMO_AMBIENT_BETA = atmoAmbientBeta;
cf.ATMO_G = atmoG;
cf.ATMO_HEIGHT_RAY = atmoHeightRay;
cf.ATMO_HEIGHT_MIE = atmoHeightMie;
cf.ATMO_HEIGHT_ABSORPTION = atmoHeightAbsorption;
cf.ATMO_ABSORPTION_FALLOFF = atmoAbsorptionFalloff;
cf.ATMO_DENSITY_STEPS = atmoDensitySteps;
cf.ATMO_LIGHT_STEPS = atmoLightSteps;   
cf.ATMO_SKYLIGHT_MULTIPLIER = atmoSkylightMultiplier;
//Cloud parameters
cf.CLOUD_COLOR = cloudColor;
cf.CLOUD_HEIGHT = cloudHeight;
cf.CLOUD_THICKNESS = cloudThickness;
cf.CLOUD_STEP_SCALE = cloudStepScale;
cf.CLOUD_DENSITY_STEPS = cloudDensitySteps;
cf.CLOUD_LIGHT_STEPS = cloudLightSteps;
cf.CLOUD_MIN_RADIUS = cf.PLANET_RADIUS + cf.CLOUD_HEIGHT;
cf.CLOUD_MAX_RADIUS = cf.CLOUD_MIN_RADIUS + cf.CLOUD_THICKNESS;
cf.CLOUD_ABSORPTION_TO_CLOUD = cloudAbsorptionToCloud;
cf.CLOUD_ABSORPTION_TO_LIGHT = cloudAbsorptionToLight;
cf.CLOUD_DARKNESS_THRESHOLD = cloudDarknessThreshold;
cf.CLOUD_G = cloudG;
cf.CLOUD_AMBIENT = cloudAmbient;
cf.CLOUD_SHADOW_SOFTNESS = cloudShadowSoftness;
cf.CLOUD_SCATTER_FADE = cloudScatterFade;
//Cloud noise parameters
cf.CLOUD_HEIGHT_FALLOFF = cloudHeightFalloff;
cf.CLOUD_DENSITY_FALLOFF = cloudDensityFalloff;
cf.CLOUD_DENSITY_FLOOR = cloudDensityFloor;
cf.CLOUD_NOISE_SCALE = cloudNoiseScale;
cf.CLOUD_NOISE_OFFSET = cloudNoiseOffset;
cf.CLOUD_DISTORTION_SCALE_WEIGHT = cloudDistortionScaleWeight;
cf.CLOUD_SHAPE_SCALE_WEIGHT = cloudShapeScaleWeight;
cf.CLOUD_DETAIL_SCALE_WEIGHT = cloudDetailScaleWeight;
cf.CLOUD_MICRO_SCALE_WEIGHT = cloudMicroScaleWeight;
//Cloud textures
cf.CLOUD_VOLUME_TEXTURE = cloudVolumeTexture;
cf.CLOUD_VOLUME_TEXTURE_SAMPLER = cloudVolumeTextureSampler;
cf.CLOUD_BLUE_NOISE_TEXTURE = cloudBlueNoiseTexture;
cf.CLOUD_BLUE_NOISE_TEXTURE_SAMPLER = cloudBlueNoiseTextureSampler;

return cf.MainImage();

/*
Cloud shader partially adapted from Sebastian Lague's video: https://www.youtube.com/watch?v=4QOcCGI6xOU

Atmosphere shader adapted from Shadertoy, atmosphere shader liscense:
MIT License

Copyright (c) 2019 Dimas Leenman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/