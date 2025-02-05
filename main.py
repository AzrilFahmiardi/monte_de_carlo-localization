import pygame
import math
import random
import logging
import os
from datetime import datetime
from statistics import mean, stdev
from robot import Robot
from map_simulation import MapSimulation
from monte_carlo import MonteCarloLocalization, Particle

# Setup logging
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
log_filename = os.path.join(log_folder, f'mcl_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def calculate_motion_delta(old_x, old_y, old_angle, new_x, new_y, new_angle):
    """Calculate motion delta between two poses"""
    delta_x = new_x - old_x
    delta_y = new_y
    delta_theta = new_angle - old_angle
    while delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    while delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    return delta_x, delta_y, delta_theta

def calculate_particle_statistics(particles, robot):
    """Calculate statistical information about particles"""
    if not particles:
        return None
        
    # Calculate mean position
    mean_x = mean(p.x for p in particles)
    mean_y = mean(p.y for p in particles)
    mean_theta = mean(p.theta for p in particles)
    
    # Calculate standard deviations
    std_x = stdev(p.x for p in particles)
    std_y = stdev(p.y for p in particles)
    std_theta = stdev(p.theta for p in particles)
    
    # Calculate error from true position
    position_error = math.sqrt((mean_x - robot.x)**2 + (mean_y - robot.y)**2)
    angle_error = abs(mean_theta - robot.angle)
    while angle_error > math.pi:
        angle_error -= 2 * math.pi
    
    # Calculate particle spread (uncertainty)
    uncertainty = math.sqrt(std_x**2 + std_y**2)
    
    return {
        'estimated_x': mean_x,
        'estimated_y': mean_y,
        'estimated_theta': mean_theta,
        'position_error': position_error,
        'angle_error': math.degrees(angle_error),
        'uncertainty': uncertainty,
        'std_x': std_x,
        'std_y': std_y,
        'std_theta': math.degrees(std_theta)
    }

def draw_particles(screen, particles, robot, color=(255, 0, 255), size=2):
    """Draw particles on screen with offset visualization and statistics"""
    if not particles:
        return None
        
    stats = calculate_particle_statistics(particles, robot)
    mean_x = stats['estimated_x']
    mean_y = stats['estimated_y']
    
    # Draw offset circle
    offset_radius = stats['position_error']
    pygame.draw.circle(screen, (0, 255, 0), (int(robot.x), int(robot.y)), 
                      int(offset_radius), 1)
    
    # Draw mean position indicator
    pygame.draw.circle(screen, (0, 255, 0), (int(mean_x), int(mean_y)), 5, 1)
    
    # Draw particles
    for p in particles:
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), size)
        end_x = p.x + size * 2 * math.cos(p.theta)
        end_y = p.y + size * 2 * math.sin(p.theta)
        pygame.draw.line(screen, color, (p.x, p.y), (end_x, end_y), 1)
        
    # Draw line between robot and mean particle position
    pygame.draw.line(screen, (0, 255, 0), 
                    (robot.x, robot.y), 
                    (mean_x, mean_y), 1)
    
    # Draw statistics on screen
    font = pygame.font.Font(None, 24)
    stats_texts = [
        f"Estimated Position: ({stats['estimated_x']:.1f}, {stats['estimated_y']:.1f})",
        f"Estimated Heading: {math.degrees(stats['estimated_theta'])::.1f}°",
        f"Position Error: {stats['position_error']:.1f}px",
        f"Angle Error: {stats['angle_error']:.1f}°",
        f"Uncertainty: {stats['uncertainty']:.1f}px"
    ]
    
    for i, text in enumerate(stats_texts):
        surface = font.render(text, True, (255, 255, 255))
        screen.blit(surface, (10, 10 + i * 25))
    
    return stats

def random_field_position(width, height):
    """Generate random position within field boundaries"""
    margin = 20
    x = random.uniform(margin, width - margin)
    y = random.uniform(margin, height - margin)
    theta = random.uniform(0, 2 * math.pi)
    return x, y, theta

def log_statistics(stats, frame_count):
    """Log particle filter statistics"""
    if stats:
        logging.info(
            f"Frame {frame_count} - "
            f"Estimated Position: ({stats['estimated_x']:.2f}, {stats['estimated_y']:.2f}), "
            f"Heading: {math.degrees(stats['estimated_theta']):.2f}° | "
            f"Errors - Position: {stats['position_error']:.2f}px, "
            f"Angle: {stats['angle_error']:.2f}° | "
            f"Uncertainty: {stats['uncertainty']:.2f}px"
        )

def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # Initialize map, robot, and particle filter
    map_simulation = MapSimulation(width, height, 20)
    robot = Robot(400, 300, 0)  # Initial robot position
    mcl = MonteCarloLocalization(
        num_particles=2000, 
        width=width, 
        height=height,
        robot_x=robot.x,
        robot_y=robot.y,
        robot_theta=robot.angle
    )
    
    frame_count = 0
    running = True
    while running:
        frame_count += 1
        screen.fill((0, 0, 0))
        map_simulation.draw_map(screen)

        # Handle robot movement and controls
        keys = pygame.key.get_pressed()
        old_x, old_y, old_angle = robot.x, robot.y, robot.angle
        robot.move(keys)

        # Calculate motion update
        if any([keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d]]):
            delta_x, delta_y, delta_theta = calculate_motion_delta(
                old_x, old_y, old_angle,
                robot.x, robot.y, robot.angle
            )
            mcl.motion_update(delta_x, delta_y, delta_theta)

        # Get LIDAR measurements and update particle filter
        lidar_rays = robot.lidar(screen, map_simulation)
        map_lines = [(line['start'], line['end']) for line in map_simulation.field_lines]
        mcl.sensor_update(robot, map_lines)
        mcl.resample()

        # Draw particles, robot, and get statistics
        stats = draw_particles(screen, mcl.particles, robot)
        if stats and frame_count % 30 == 0:  # Log every 30 frames
            log_statistics(stats, frame_count)
            
        robot.draw(screen)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset robot to center and reinitialize particles around it
                    robot = Robot(400, 300, 0)
                    mcl.reinitialize_particles(robot.x, robot.y, robot.angle)
                    
                elif event.key == pygame.K_i:
                    # Teleport robot to random position
                    x, y, theta = random_field_position(width, height)
                    robot = Robot(x, y, theta)
                    # Don't reinitialize particles - let them converge to new position

    pygame.quit()

if __name__ == "__main__":
    main()