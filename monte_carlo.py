import random
import math
import numpy as np

class Particle:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0

class MonteCarloLocalization:
    def __init__(self, num_particles, width, height):
        self.num_particles = num_particles
        self.width = width
        self.height = height
        # Configure key parameters
        self.min_particles = num_particles // 4
        self.recovery_alpha = 0.3  # Increased from 0.1 for better recovery
        self.effective_particles_ratio = 0.5
        # Motion model noise parameters
        self.alpha1 = 0.2  # Increased noise parameters
        self.alpha2 = 0.2
        self.alpha3 = 0.3
        self.alpha4 = 0.2
        self.particles = [self.random_particle() for _ in range(num_particles)]
        
        # Add kidnapping detection parameters
        self.avg_weight_history = []
        self.weight_history_size = 10

    def detect_kidnapping(self):
        """Detect if robot might have been kidnapped based on particle weights"""
        if len(self.avg_weight_history) >= self.weight_history_size:
            current_avg = sum(p.weight for p in self.particles) / len(self.particles)
            historical_avg = sum(self.avg_weight_history) / len(self.avg_weight_history)
            
            # If current weights are significantly lower than historical average
            if current_avg < historical_avg * self.kidnapping_threshold:
                return True
                
            # Update history
            self.avg_weight_history.pop(0)
            self.avg_weight_history.append(current_avg)
        return False
        
    def random_particle(self):
        # Generate particles within known field boundaries (white lines)
        margin = 50  # Increased margin for better boundary handling
        x = random.uniform(100 + margin, 700 - margin)  # Field boundaries from map
        y = random.uniform(100 + margin, 500 - margin)
        theta = random.uniform(0, 2 * math.pi)
        return Particle(x, y, theta)

    def handle_kidnapping(self):
        """Recovery behavior when kidnapping is detected"""
        # Keep some best particles
        num_keep = self.num_particles // 10
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        best_particles = sorted_particles[:num_keep]
        
        # Add many random particles
        num_random = self.num_particles - num_keep
        random_particles = [self.random_particle() for _ in range(num_random)]
        
        # Combine and normalize weights
        self.particles = best_particles + random_particles
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight

    def motion_update(self, delta_x, delta_y, delta_theta):
        """Improved motion model with better noise modeling"""
        translation = math.sqrt(delta_x**2 + delta_y**2)
        
        for particle in self.particles:
            # Calculate noisy motion parameters
            trans_noise1 = random.gauss(0, self.alpha3 * translation + 
                                      self.alpha4 * abs(delta_theta))
            trans_noise2 = random.gauss(0, self.alpha3 * translation + 
                                      self.alpha4 * abs(delta_theta))
            rot_noise = random.gauss(0, self.alpha1 * abs(delta_theta) + 
                                   self.alpha2 * translation)
            
            # Apply noisy motion
            noisy_translation = translation + trans_noise1
            noisy_rotation = delta_theta + rot_noise
            
            # Update particle pose
            particle.x += noisy_translation * math.cos(particle.theta)
            particle.y += noisy_translation * math.sin(particle.theta)
            particle.theta = (particle.theta + noisy_rotation) % (2 * math.pi)
            
            # Bounce particles off boundaries instead of clamping
            self.handle_boundaries(particle)

    def handle_boundaries(self, particle):
        """Improved boundary handling with bounce behavior"""
        margin = 20
        
        # Check x boundaries
        if particle.x < 100 + margin:
            particle.x = 100 + margin + (100 + margin - particle.x)
            particle.theta = math.pi - particle.theta
        elif particle.x > 700 - margin:
            particle.x = 700 - margin - (particle.x - (700 - margin))
            particle.theta = math.pi - particle.theta
            
        # Check y boundaries    
        if particle.y < 100 + margin:
            particle.y = 100 + margin + (100 + margin - particle.y)
            particle.theta = -particle.theta
        elif particle.y > 500 - margin:
            particle.y = 500 - margin - (particle.y - (500 - margin))
            particle.theta = -particle.theta

    def sensor_update(self, robot, map_lines):
        """Enhanced sensor model with kidnapping detection"""
        robot_measurements = self.get_robot_measurements(robot, map_lines)
        
        max_weight = 0
        for particle in self.particles:
            particle_measurements = self.get_particle_measurements(particle, map_lines)
            likelihood = self.compute_measurement_likelihood(
                robot_measurements, particle_measurements)
            particle.weight = likelihood
            max_weight = max(max_weight, likelihood)
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        
        # Check for kidnapping
        if self.detect_kidnapping():
            self.handle_kidnapping()
        else:
            self.check_and_recover()

    def get_robot_measurements(self, robot, map_lines):
        """Get actual measurements from robot's LIDAR"""
        measurements = []
        angles = np.linspace(-math.pi/6, math.pi/6, 8)  # 8 rays spread over 60 degrees
        
        for angle in angles:
            ray_angle = robot.angle + angle
            dist = self.cast_ray(robot.x, robot.y, ray_angle, map_lines)
            measurements.append((angle, dist))
            
        return measurements

    def get_particle_measurements(self, particle, map_lines):
        """Get expected measurements for particle position"""
        measurements = []
        angles = np.linspace(-math.pi/6, math.pi/6, 8)
        
        for angle in angles:
            ray_angle = particle.theta + angle
            dist = self.cast_ray(particle.x, particle.y, ray_angle, map_lines)
            measurements.append((angle, dist))
            
        return measurements

    def compute_measurement_likelihood(self, robot_meas, particle_meas):
        """Compute measurement likelihood with improved noise model"""
        sigma = 10.0  # Measurement noise parameter
        likelihood = 1.0
        
        for (r_angle, r_dist), (p_angle, p_dist) in zip(robot_meas, particle_meas):
            # Gaussian error model
            error = (r_dist - p_dist) ** 2
            likelihood *= math.exp(-error / (2 * sigma ** 2))
            
        return likelihood

    def check_and_recover(self):
        """Check for particle filter degeneracy and recover if needed"""
        # Calculate effective number of particles
        neff = 1.0 / sum(p.weight ** 2 for p in self.particles)
        
        if neff < self.num_particles * self.effective_particles_ratio:
            # Add random particles for recovery
            num_random = int(self.num_particles * self.recovery_alpha)
            sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
            
            # Keep best particles and add random ones
            self.particles = sorted_particles[:-num_random]
            self.particles.extend([self.random_particle() for _ in range(num_random)])

    def resample(self):
        """Improved resampling with systematic resampling"""
        new_particles = []
        
        # Systematic resampling
        positions = (np.random.random() + np.arange(self.num_particles)) / self.num_particles
        cumulative_sum = np.cumsum([p.weight for p in self.particles])
        
        i = 0
        for position in positions:
            while cumulative_sum[i] < position:
                i += 1
            # Add noise to resampled particles to prevent sample impoverishment
            new_particle = Particle(
                self.particles[i].x + random.gauss(0, 0.5),
                self.particles[i].y + random.gauss(0, 0.5),
                (self.particles[i].theta + random.gauss(0, 0.05)) % (2 * math.pi)
            )
            new_particles.append(new_particle)
            
        self.particles = new_particles

    def cast_ray(self, x, y, angle, map_lines):
        """Cast a ray and return distance to nearest line intersection"""
        ray_length = 200  # Maximum ray length
        ray_end_x = x + ray_length * math.cos(angle)
        ray_end_y = y + ray_length * math.sin(angle)
        
        min_dist = ray_length
        
        # Check intersection with each line segment
        for line in map_lines:
            start, end = line
            intersection = self.line_intersection(
                (x, y), (ray_end_x, ray_end_y),
                start, end
            )
            if intersection:
                dist = math.sqrt((x - intersection[0])**2 + (y - intersection[1])**2)
                min_dist = min(min_dist, dist)
                
        return min_dist

    def line_intersection(self, p1, p2, p3, p4):
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
            
        return None