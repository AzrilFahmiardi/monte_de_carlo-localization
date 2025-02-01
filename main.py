import pygame
import math
import random
from robot import Robot
from map_simulation import MapSimulation
from monte_carlo import MonteCarloLocalization, Particle

def calculate_motion_delta(old_x, old_y, old_angle, new_x, new_y, new_angle):
    """Calculate motion delta between two poses"""
    delta_x = new_x - old_x
    delta_y = new_y - old_y
    delta_theta = new_angle - old_angle
    while delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    while delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    return delta_x, delta_y, delta_theta

def draw_particles(screen, particles, robot, color=(255, 0, 255), size=2):
    """Draw particles on screen with offset visualization"""
    # Calculate mean position of particles
    if not particles:
        return
        
    mean_x = sum(p.x for p in particles) / len(particles)
    mean_y = sum(p.y for p in particles) / len(particles)
    
    # Draw offset circle
    offset_radius = math.sqrt((mean_x - robot.x)**2 + (mean_y - robot.y)**2)
    pygame.draw.circle(screen, (0, 255, 0), (int(robot.x), int(robot.y)), 
                      int(offset_radius), 1)
    
    # Draw mean position indicator
    pygame.draw.circle(screen, (0, 255, 0), (int(mean_x), int(mean_y)), 5, 1)
    
    # Draw particles
    for p in particles:
        # Make particles more visible
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), size)
        # Draw direction line
        end_x = p.x + size * 2 * math.cos(p.theta)
        end_y = p.y + size * 2 * math.sin(p.theta)
        pygame.draw.line(screen, color, (p.x, p.y), (end_x, end_y), 1)
        
    # Draw line between robot and mean particle position
    pygame.draw.line(screen, (0, 255, 0), 
                    (robot.x, robot.y), 
                    (mean_x, mean_y), 1)

def random_field_position(width, height):
    """Generate random position within field boundaries"""
    margin = 20
    x = random.uniform(margin, width - margin)
    y = random.uniform(margin, height - margin)
    theta = random.uniform(0, 2 * math.pi)
    return x, y, theta

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
    
    prev_x, prev_y, prev_angle = robot.x, robot.y, robot.angle

    running = True
    while running:
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

        # Draw particles and robot
        draw_particles(screen, mcl.particles, robot)  # Menambahkan parameter robot
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