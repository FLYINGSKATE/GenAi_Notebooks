# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "llm-deployment"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "llm_api" {
  name                 = "llm-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "llm_cluster" {
  name = "llm-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "llm_api" {
  family                   = "llm-api-${var.environment}"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = var.task_cpu
  memory                  = var.task_memory
  execution_role_arn      = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name         = "llm-api"
      image        = "${aws_ecr_repository.llm_api.repository_url}:latest"
      essential    = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "MODEL_PATH"
          value = var.model_path
        },
        {
          name  = "SECRET_KEY"
          value = var.secret_key
        },
        {
          name  = "ENVIRONMENT"
          value = var.environment
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/llm-api-${var.environment}"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "llm_api" {
  name            = "llm-api-${var.environment}"
  cluster         = aws_ecs_cluster.llm_cluster.id
  task_definition = aws_ecs_task_definition.llm_api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.llm_api.arn
    container_name   = "llm-api"
    container_port   = 8000
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_task_execution_role,
    aws_lb_listener.https_forward,
  ]
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "llm-api-sg-${var.environment}"
  description = "Security group for LLM API tasks"
  vpc_id      = var.vpc_id

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Application Load Balancer
resource "aws_lb" "llm_api" {
  name               = "llm-api-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = var.public_subnets

  enable_deletion_protection = false
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name        = "llm-api-alb-sg-${var.environment}"
  description = "Security group for ALB"
  vpc_id      = var.vpc_id

  ingress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Target Group
resource "aws_lb_target_group" "llm_api" {
  name        = "llm-api-tg-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path                = "/health"
    port                = "traffic-port"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }
}

# HTTPS Listener
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.llm_api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.llm_api.arn
  }
}

# IAM Role for ECS Tasks
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs-task-execution-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Task Role
resource "aws_iam_role" "ecs_task_role" {
  name = "ecs-task-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Outputs
output "alb_dns_name" {
  value = aws_lb.llm_api.dns_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.llm_api.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.llm_cluster.name
}

output "ecs_service_name" {
  value = aws_ecs_service.llm_api.name
}
