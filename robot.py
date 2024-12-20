import pygame
import math

class Robot:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # Direction in radians
        self.radius = 10
        self.lidar_range = 200  # Maximum LIDAR range
        self.lidar_angle_range = 30  # LIDAR angle range in degrees
        self.num_lidar_rays = 50  # Number of LIDAR rays
        self.line_detection_threshold = 3  # Minimum pixels to consider as line detection

    def move(self, keys):
        speed = 8
        turn_speed = 0.15
        
        if keys[pygame.K_w]:  # Move forward
            self.x += speed * math.cos(self.angle)
            self.y += speed * math.sin(self.angle)
        if keys[pygame.K_s]:  # Move backward
            self.x -= speed * math.cos(self.angle)
            self.y -= speed * math.sin(self.angle)
        if keys[pygame.K_a]:  # Turn left
            self.angle -= turn_speed
        if keys[pygame.K_d]:  # Turn right
            self.angle += turn_speed

    def draw(self, screen):
        # Draw transparent circle with stroke
        circle_surface = pygame.Surface((self.radius * 2 + 2, self.radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 255, 40), (self.radius + 1, self.radius + 1), self.radius)  # Filled transparent circle
        pygame.draw.circle(circle_surface, (0, 0, 255, 255), (self.radius + 1, self.radius + 1), self.radius, 2)  # Stroke
        screen.blit(circle_surface, (int(self.x - self.radius - 1), int(self.y - self.radius - 1)))

        # Draw direction indicator
        line_length = 20
        pygame.draw.line(screen, (255, 0, 0), 
                        (self.x, self.y),
                        (self.x + line_length * math.cos(self.angle),
                         self.y + line_length * math.sin(self.angle)), 2)

    def draw_lidar_fov(self, screen):
        # Draw LIDAR field of view
        half_angle = math.radians(self.lidar_angle_range / 2)
        start_angle = self.angle - half_angle
        end_angle = self.angle + half_angle
        
        # Draw FOV boundaries
        pygame.draw.line(screen, (100, 100, 100),
                        (self.x, self.y),
                        (self.x + self.lidar_range * math.cos(start_angle),
                         self.y + self.lidar_range * math.sin(start_angle)), 1)
        pygame.draw.line(screen, (100, 100, 100),
                        (self.x, self.y),
                        (self.x + self.lidar_range * math.cos(end_angle),
                         self.y + self.lidar_range * math.sin(end_angle)), 1)

    def lidar(self, screen, map_simulation):
        lidar_rays = []
        half_angle = math.radians(self.lidar_angle_range / 2)
        
        # Draw LIDAR field of view
        self.draw_lidar_fov(screen)

        # Cast rays within the field of view
        for i in range(self.num_lidar_rays):
            ray_angle = self.angle - half_angle + (i * 2 * half_angle / (self.num_lidar_rays - 1))
            
            # Sample multiple points along each ray to detect line width
            line_detected = False
            detection_points = []
            
            for dist in range(1, self.lidar_range, 2):
                ray_x = self.x + dist * math.cos(ray_angle)
                ray_y = self.y + dist * math.sin(ray_angle)
                
                # Check for white line detection
                if map_simulation.is_white_line(ray_x, ray_y):
                    detection_points.append((ray_x, ray_y))
                    
                    # If we have enough consecutive detections, consider it a valid line
                    if len(detection_points) >= self.line_detection_threshold:
                        line_detected = True
                        # Calculate average position of detection points
                        avg_x = sum(p[0] for p in detection_points) / len(detection_points)
                        avg_y = sum(p[1] for p in detection_points) / len(detection_points)
                        lidar_rays.append((avg_x, avg_y))
                        # Draw detection ray
                        pygame.draw.line(screen, (255, 0, 0), 
                                       (self.x, self.y), 
                                       (avg_x, avg_y), 2)
                        break
                else:
                    detection_points = []  # Reset detection points if line is lost

        return lidar_rays